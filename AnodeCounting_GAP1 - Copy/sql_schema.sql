CREATE TABLE dbo.AnodeCountDaily (
    Id BIGINT IDENTITY(1,1) PRIMARY KEY,
    CountDate DATE NOT NULL,
    CameraName NVARCHAR(100) NOT NULL,
    HoleCount INT NOT NULL DEFAULT 0,
    AnodeCount INT NOT NULL DEFAULT 0,
    LastEventTime DATETIME2 NULL,
    UpdatedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT UQ_AnodeCountDaily UNIQUE (CountDate, CameraName)
);

CREATE TABLE dbo.HoleCrossingEvents (
    Id BIGINT IDENTITY(1,1) PRIMARY KEY,
    EventTime DATETIME2 NOT NULL,
    CountDate AS CAST(EventTime AS DATE) PERSISTED,
    CameraName NVARCHAR(100) NOT NULL,
    TrackId INT NOT NULL,
    Cx FLOAT NOT NULL,
    Cy FLOAT NOT NULL,
    Confidence FLOAT NULL,
    FrameId BIGINT NULL
);
GO

CREATE OR ALTER PROCEDURE dbo.spUpsertAnodeDaily
    @CountDate DATE,
    @CameraName NVARCHAR(100),
    @HoleCount INT,
    @AnodeCount INT,
    @LastEventTime DATETIME2
AS
BEGIN
    SET NOCOUNT ON;

    MERGE dbo.AnodeCountDaily AS T
    USING (SELECT @CountDate AS CountDate, @CameraName AS CameraName) AS S
    ON T.CountDate = S.CountDate AND T.CameraName = S.CameraName
    WHEN MATCHED THEN
        UPDATE SET
            HoleCount = @HoleCount,
            AnodeCount = @AnodeCount,
            LastEventTime = @LastEventTime,
            UpdatedAt = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN
        INSERT (CountDate, CameraName, HoleCount, AnodeCount, LastEventTime)
        VALUES (@CountDate, @CameraName, @HoleCount, @AnodeCount, @LastEventTime);
END
GO