/*
Migration 002 - daily totals derived from events; UTC timestamps; indexes

Problem fixed: dbo.AnodeCountDaily was overwritten with the service's
in-memory counters. Counters were never reset at midnight (every day after
the first included all earlier days) and were reset to zero on restart
(the day's total was erased).

- dbo.spRefreshAnodeDaily recomputes one day's row from the event tables,
  so the row is always consistent with the events, whatever the service
  restarts or however late events are delivered after an outage.
- EventTimeUtc / AnodeTimeUtc store the same instant as EventTime / AnodeTime
  in UTC. The existing local-time columns are unchanged (Power BI shift and
  date logic keeps working).
- Indexes for per-camera, per-day queries (daily refresh and dashboards).

Additive and re-runnable. dbo.spUpsertAnodeDaily is left in place so the
previous service version still works if a rollback is needed.
Run 001_event_uid.sql first.
*/
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF COL_LENGTH('dbo.HoleCrossingEvents', 'EventTimeUtc') IS NULL
    ALTER TABLE dbo.HoleCrossingEvents ADD EventTimeUtc DATETIME2 NULL;
GO

IF COL_LENGTH('dbo.AnodeEvents', 'AnodeTimeUtc') IS NULL
    ALTER TABLE dbo.AnodeEvents ADD AnodeTimeUtc DATETIME2 NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_HoleCrossingEvents_Camera_Date' AND object_id = OBJECT_ID('dbo.HoleCrossingEvents'))
    CREATE NONCLUSTERED INDEX IX_HoleCrossingEvents_Camera_Date
        ON dbo.HoleCrossingEvents(CameraName, CountDate) INCLUDE (EventTime);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'IX_AnodeEvents_Camera_Time' AND object_id = OBJECT_ID('dbo.AnodeEvents'))
    CREATE NONCLUSTERED INDEX IX_AnodeEvents_Camera_Time
        ON dbo.AnodeEvents(CameraName, AnodeTime);
GO

CREATE OR ALTER PROCEDURE dbo.spRefreshAnodeDaily
    @CountDate DATE,
    @CameraName NVARCHAR(100)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @DayStart DATETIME2 = CAST(@CountDate AS DATETIME2);
    DECLARE @DayEnd DATETIME2 = DATEADD(DAY, 1, @DayStart);
    DECLARE @Holes INT, @Anodes INT, @LastHole DATETIME2, @LastAnode DATETIME2, @Last DATETIME2;

    SELECT @Holes = COUNT(*), @LastHole = MAX(EventTime)
    FROM dbo.HoleCrossingEvents
    WHERE CameraName = @CameraName AND CountDate = @CountDate;

    SELECT @Anodes = COUNT(*), @LastAnode = MAX(AnodeTime)
    FROM dbo.AnodeEvents
    WHERE CameraName = @CameraName AND AnodeTime >= @DayStart AND AnodeTime < @DayEnd;

    SET @Last = CASE WHEN @LastHole IS NULL OR @LastAnode > @LastHole THEN @LastAnode ELSE @LastHole END;

    MERGE dbo.AnodeCountDaily WITH (HOLDLOCK) AS T
    USING (SELECT @CountDate AS CountDate, @CameraName AS CameraName) AS S
    ON T.CountDate = S.CountDate AND T.CameraName = S.CameraName
    WHEN MATCHED THEN
        UPDATE SET
            HoleCount = @Holes,
            AnodeCount = @Anodes,
            LastEventTime = @Last,
            UpdatedAt = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN
        INSERT (CountDate, CameraName, HoleCount, AnodeCount, LastEventTime)
        VALUES (@CountDate, @CameraName, @Holes, @Anodes, @Last);
END
GO
