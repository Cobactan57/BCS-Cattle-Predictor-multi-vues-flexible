insert into storage.buckets (id, name, public)
values ('bcs-models', 'bcs-models', false)
on conflict (id) do update set public = false;
