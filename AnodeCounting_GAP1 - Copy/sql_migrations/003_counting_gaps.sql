/*
Migration 003 - counting gaps and processing lag

dbo.CountingGaps records every interval in which the counter could NOT see
the belt reliably, so a missing anode can be explained instead of silently
disappearing from the totals. Reasons:
  SOURCE_DISCONNECTED  camera stream lost / reconnecting
  SOURCE_STALL         camera connected but delivered no frames
  PROCESSING_GAP       frames came too far apart to track holes reliably
  PIPELINE_RESTART     counting pipeline recovered from an error
  SERVICE_DOWN         the service was not running (detected at next start)
  LOW_VISIBILITY       fumes: contrast low and no hole detected confidently (visibility.py)

ProcessingLagMs on the event tables is the time from frame capture to the
event being recorded, i.e. how far the count was behind the belt.

Additive and re-runnable. Run 001 and 002 first.
*/
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF OBJECT_ID('dbo.CountingGaps', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.CountingGaps (
        GapId BIGINT IDENTITY(1,1) PRIMARY KEY,
        EventUid UNIQUEIDENTIFIER NOT NULL,
        CameraName NVARCHAR(100) NOT NULL,
        StartTime DATETIME2 NOT NULL,      -- plant local time, like EventTime
        EndTime DATETIME2 NOT NULL,
        StartTimeUtc DATETIME2 NOT NULL,
        EndTimeUtc DATETIME2 NOT NULL,
        DurationMs BIGINT NOT NULL,
        Reason NVARCHAR(32) NOT NULL,
        Details NVARCHAR(400) NULL,
        CreatedAtUtc DATETIME2 NOT NULL CONSTRAINT DF_CountingGaps_CreatedAtUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_CountingGaps_EventUid UNIQUE (EventUid)
    );
    CREATE NONCLUSTERED INDEX IX_CountingGaps_Camera_Start ON dbo.CountingGaps(CameraName, StartTime);
END
GO

IF COL_LENGTH('dbo.HoleCrossingEvents', 'ProcessingLagMs') IS NULL
    ALTER TABLE dbo.HoleCrossingEvents ADD ProcessingLagMs INT NULL;
GO

IF COL_LENGTH('dbo.AnodeEvents', 'ProcessingLagMs') IS NULL
    ALTER TABLE dbo.AnodeEvents ADD ProcessingLagMs INT NULL;
GO
