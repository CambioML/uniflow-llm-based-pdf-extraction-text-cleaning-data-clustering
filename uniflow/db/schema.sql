--  ExpandReduceFlow psql schema

drop type if exists value_type;
-- add more type if needed
create type value_type as enum('int', 'str', 'float');

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
    node  text references nodes(nname),
    next  text references nodes(nname),
    primary key (node, next)
);
