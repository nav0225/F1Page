/*
Migration 001 - unique event ids (idempotent delivery)

Adds an EventUid column to the event tables plus a filtered unique index, so
the service can retry a write after a network error without ever creating a
duplicate row.

- Additive only: no existing column, row or object is modified.
- Safe to run more than once.
- The previous service version keeps working after this migration (it simply
  leaves EventUid NULL).

Run order: sql_schema.sql (fresh installs only), then sql_migrations/*.sql in
number order. Apply on a test copy of the database first.
*/
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF OBJECT_ID('dbo.AnodeEvents', 'U') IS NULL
BEGIN
    -- Fresh installs only. Production already has this table, but its DDL was
    -- never committed to the repository; export it and reconcile with this
    -- definition (columns below are the ones the service writes).
    CREATE TABLE dbo.AnodeEvents (
        AnodeId BIGINT IDENTITY(1,1) PRIMARY KEY,
        AnodeTime DATETIME2 NOT NULL,
        CameraName NVARCHAR(100) NOT NULL,
        HoleStartNo INT NULL,
        HoleEndNo INT NULL
    );
END
GO

IF COL_LENGTH('dbo.HoleCrossingEvents', 'EventUid') IS NULL
    ALTER TABLE dbo.HoleCrossingEvents ADD EventUid UNIQUEIDENTIFIER NULL;
GO

IF COL_LENGTH('dbo.AnodeEvents', 'EventUid') IS NULL
    ALTER TABLE dbo.AnodeEvents ADD EventUid UNIQUEIDENTIFIER NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'UX_HoleCrossingEvents_EventUid' AND object_id = OBJECT_ID('dbo.HoleCrossingEvents'))
    CREATE UNIQUE NONCLUSTERED INDEX UX_HoleCrossingEvents_EventUid
        ON dbo.HoleCrossingEvents(EventUid) WHERE EventUid IS NOT NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'UX_AnodeEvents_EventUid' AND object_id = OBJECT_ID('dbo.AnodeEvents'))
    CREATE UNIQUE NONCLUSTERED INDEX UX_AnodeEvents_EventUid
        ON dbo.AnodeEvents(EventUid) WHERE EventUid IS NOT NULL;
GO
