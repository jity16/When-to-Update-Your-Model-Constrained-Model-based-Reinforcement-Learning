const express = require("express");
const body_parser = require("body-parser");
const path = require("path");
const fs = require("fs");
const config = require("./config");
const utils = require("./utils");
const services = require("./services");

const app = express();

app.set("view engine", "ejs");

app.use("/static", express.static(path.join(__dirname, "static")));

app.use((req, res, next) => {
    res.locals.title = "";
    res.locals.brand = config.brand;
    
    const current_timestamp = utils.current_timestamp();
    res.locals.current_timestamp = current_timestamp;
    res.locals.current_datetime = utils.timestamp_to_datetime(current_timestamp);
    
    res.locals.req = req;
    res.locals.res = res;
    res.locals.config = config;
    
    res.locals.utils = utils;
    res.locals.fs = fs;
    
    next();
});

app.use(body_parser.urlencoded({ extended: true }));

require("./pages")(app);

const args = process.argv.slice(2);
if (args.length != 2) {
    console.log(`Usage: node ${process.argv[1]} <hostname> <port>`);
    process.exit(1);
}

const start_server = async () => {
    try {
        await services.query_all();
        
        const hostname = args[0];
        const port = args[1];
        
        app.listen(port, hostname, () => {
            console.log(`Listening on ${hostname}:${port}`);
        });
    } catch (e) {
        console.error(e);
    }
};

start_server();
