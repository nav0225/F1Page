/*
Migration 004 - rejected hole crossings

A rejected crossing is a tracked object that crossed the counting line but
never had enough confident detections (COUNT_MIN_HITS at COUNT_MIN_CONF) to
be counted as a hole. Typical causes: fumes, partial occlusion, or a false
detection. They are NOT included in any total; they are kept so that a
missing hole can be investigated instead of disappearing without a trace.

Separate table so HoleCrossingEvents keeps meaning "counted holes" for every
existing consumer. Additive and re-runnable. Run 001-003 first.
*/
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF OBJECT_ID('dbo.RejectedHoleCrossings', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.RejectedHoleCrossings (
        Id BIGINT IDENTITY(1,1) PRIMARY KEY,
        EventUid UNIQUEIDENTIFIER NOT NULL,
        EventTime DATETIME2 NOT NULL,      -- plant local time of the crossing
        EventTimeUtc DATETIME2 NOT NULL,
        CameraName NVARCHAR(100) NOT NULL,
        TrackId INT NOT NULL,
        Cx FLOAT NOT NULL,
        Cy FLOAT NOT NULL,
        MaxConfidence FLOAT NULL,
        StrongHits INT NOT NULL,
        Reason NVARCHAR(32) NOT NULL,
        FrameId BIGINT NULL,
        CreatedAtUtc DATETIME2 NOT NULL CONSTRAINT DF_RejectedHoleCrossings_CreatedAtUtc DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_RejectedHoleCrossings_EventUid UNIQUE (EventUid)
    );
    CREATE NONCLUSTERED INDEX IX_RejectedHoleCrossings_Camera_Time ON dbo.RejectedHoleCrossings(CameraName, EventTime);
END
GO
