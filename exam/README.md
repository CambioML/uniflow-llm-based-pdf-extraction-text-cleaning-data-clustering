# Backend Interview README

## Database:
- ***Postgresql 12***
    - install postgresql.
      `sudo apt-get update`
      `yes | sudo apt-get install postgresql`

    - start the DB server.
      `sudo service postgresql start`

    - enter DB, set a password.
      `sudo -u postgres psql postgres`
      `\password postgres`
      `\t`

    - create DB for uniflow.
      `sudo -u postgres createdb uniflow -O uniflowAdmin`

    - go to *uniflow/uniflow/db*, load schema.
      `\i schema.sql` then `\q`

    - make it ready for python.
      `pip3 install psycopg2`
      `pip3 install psycopg2-binary` if import error occur.
