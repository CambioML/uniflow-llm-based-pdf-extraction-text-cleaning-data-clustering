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
    `pip3 install psycopg2` or
    `pip3 install psycopg2-binary` if import error occur.

## API:
- **POST: /api/v1/flow/expandReduce**
  - take an input and build as root node, then do expand and reduce op.
  - request format `{key: value, ...}`, example: `{"How are you?": "Fine."}`
  - respond format `{id: a five digits number}`, example: `{id: 12345}`

- **GET: /api/v1/flow/expandReduce**
  - take an id and return its status.
  - request format `{id: a five digits number}`, example: `{id: 12345}`
  - respond format `{status: str}`, example: `{status: pending}`

- **GET: /api/v1/node/value**
  - get all key value pairs from single page(max 50 rows)
  - request format `{page: integer}`, example: `{page: 1}`
  - respond format `{key: value, ...}` max 50 pairs, example: `{"How are you?": "Fine."}`
