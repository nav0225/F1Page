/*
Migration 005 - anode grouping evidence and shadow comparison

AnodeEvents gains the evidence behind each counted anode:
  GroupingMethod  MODULO (legacy: every 4th counted hole) or STRUCTURAL
                  (belt geometry, grouping.py)
  HolesSeen       number of this anode's holes that were counted
  CountStatus     STRUCTURAL only: CONFIRMED (all holes, slots read from the
                  belt pattern), PARTIAL (some holes not counted), UNCERTAIN
                  (slots inferred). NULL for MODULO = not assessed.

dbo.AnodeGroupingShadow stores the anodes produced by the grouping method
that is NOT used for counting (ANODE_GROUPING selects the counted one). They
are never part of any total; they let operations compare both methods on
real production before switching.

Additive and re-runnable. Run 001-004 first.
*/
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF COL_LENGTH('dbo.AnodeEvents', 'GroupingMethod') IS NULL
    ALTER TABLE dbo.AnodeEvents ADD GroupingMethod NVARCHAR(16) NULL;
GO
IF COL_LENGTH('dbo.AnodeEvents', 'HolesSeen') IS NULL
    ALTER TABLE dbo.AnodeEvents ADD HolesSeen TINYINT NULL;
GO
IF COL_LENGTH('dbo.AnodeEvents', 'CountStatus') IS NULL
    ALTER TABLE dbo.AnodeEvents ADD CountStatus NVARCHAR(16) NULL;
GO

IF OBJECT_ID('dbo.AnodeGroupingShadow', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.AnodeGroupingShadow (
        Id BIGINT IDENTITY(1,1) PRIMARY KEY,
        EventUid UNIQUEIDENTIFIER NOT NULL,
        AnodeTime DATETIME2 NOT NULL,      -- plant local time
        AnodeTimeUtc DATETIME2 NOT NULL,
        CameraName NVARCHAR(100) NOT NULL,
        GroupingMethod NVARCHAR(16) NOT NULL,
        HolesSeen TINYINT NOT NULL,
        CountStatus NVARCHAR(16) NULL,
        Slots NVARCHAR(32) NULL,           -- slot of each counted hole, e.g. '0 2 3'
        CloseReason NVARCHAR(16) NULL,
        HoleStartNo INT NULL,
        HoleEndNo INT NULL,
        CreatedAtUtc DATETIME2 NOT NULL CONSTRAINT DF_AnodeGroupingShadow_CreatedAtUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_AnodeGroupingShadow_EventUid UNIQUE (EventUid)
    );
    CREATE NONCLUSTERED INDEX IX_AnodeGroupingShadow_Camera_Time ON dbo.AnodeGroupingShadow(CameraName, AnodeTime);
END
GO
