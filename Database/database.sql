CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL
);

CREATE TABLE wishlist (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    app_name VARCHAR(255) NOT NULL
);

CREATE TABLE user_reviews (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    app_name VARCHAR(255) NOT NULL,
    rating INTEGER,
    review TEXT
);

INSERT INTO users (username, password, role)
VALUES ('admin', 'admin@123', 'admin');
select * from users;