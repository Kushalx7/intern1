FROM postgres:15

ENV POSTGRES_DB=stocks
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

COPY init.sql /docker-entrypoint-initdb.d/init.sql

EXPOSE 5432
