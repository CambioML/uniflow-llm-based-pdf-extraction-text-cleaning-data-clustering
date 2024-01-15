--  ExpandReduceFlow psql schema

drop type if exists value_type;
-- add more type if needed
create type value_type as enum('int', 'str', 'float');

drop type if exists _status;
create type _status as
    enum('pending', 'running', 'complete', 'failed');

create table if not exists nodes (
    nname       text primary key,
    is_end      boolean not null
);

create table if not exists value_dict (
    nname       text references nodes(nname),
    key         text not null,
    value       text not null,
    type        value_type,
    primary key (nname, key, value)
);

create table if not exists node_edges (
    node        text references nodes(nname),
    next        text references nodes(nname),
    primary key (node, next)
);

create table if not exists request_status (
    id          char(5) primary key,
    status      _status not null,
    _time       timestamp default current_timestamp
);

-- anthor verion of `request_status`, less row, and might helpful for the
-- problem in `request_status.py` line 26
-- create table if not exists request_status_v2 (
--     id          char(5) primary key,
--     -- below could be null, update when status change
--     pending     timestamp,
--     start       timestamp,
--     finish      timestamp,
--     cancel      timestamp
-- );
