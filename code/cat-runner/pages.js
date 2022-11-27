// Pages

const services = require("./services");
const wishlist = require("./wishlist");
const utils = require("./utils");

const index_page = async (_, res) => {
    const result = await services.query_all();
    res.locals.runs = result.runs;
    
    res.render("index");
};

const detail_page = async (req, res) => {
    const run_id = parseInt(req.params.run_id);
    res.locals.run = services.query_run(run_id);
    res.locals.stdout_content = services.read_stdout(run_id);
    res.locals.stderr_content = services.read_stderr(run_id);
    
    res.locals.run = await res.locals.run;
    res.locals.stdout_content = await res.locals.stdout_content;
    res.locals.stderr_content = await res.locals.stderr_content;
    
    res.render("detail");
};

const stdout_content = async (req, res) => {
    const run_id = parseInt(req.params.run_id);
    
    res.set("content-type", "text/plain");
    res.send(await services.read_stdout(run_id));
};

const stderr_content = async (req, res) => {
    const run_id = parseInt(req.params.run_id);
    
    res.set("content-type", "text/plain");
    res.send(await services.read_stderr(run_id));
};

const wishlist_page = async (_, res) => {
    res.locals.runs = services.query_all()
    res.locals.wishlist = wishlist.query_all();
    
    res.locals.runs = (await res.locals.runs).runs;
    res.locals.wishlist = await res.locals.wishlist;
    
    res.render("wishlist");
};

const wish_add = async (req, res) => {
    const wish = req.body;
    wish.status = "Wishing";
    wish.submit_time = utils.current_datetime();
    
    await wishlist.add(wish);
    res.send("OK");
};

const wish_run = async (req, res) => {
    const id = parseInt(req.body.id);
    try {
        await wishlist.run(id);
        await wishlist.update(id, { status: "Fulfilled" }, { status: "Wishing" });
        
        res.send("OK");
    } catch (e) {
        res.send(String(e));
    }
};

const wish_cancel = async (req, res) => {
    const id = parseInt(req.body.id);
    try {
        await wishlist.update(id, { status: "Cancelled" }, { status: "Wishing" });
        
        res.send("OK");
    } catch (e) {
        res.send(String(e));
    }
};

const tensorboard_add = async (req, res) => {
    const id = parseInt(req.body.id);
    
    const output = await utils.run_process(
        "bash",
        ["-c", `cd .. && tools/add_to_tensorboard ${id}`],
        30000,
    ).catch(e => e).then(x => x.error ? x.stderr : x.stdout);
    
    if (output.indexOf("Successfully added") != -1) {
        res.send("OK");
    } else {
        res.send(output);
    }
};

const tensorboard_del = async (req, res) => {
    const id = parseInt(req.body.id);
    
    const output = await utils.run_process(
        "bash",
        ["-c", `cd .. && tools/del_from_tensorboard ${id}`],
        30000,
    ).catch(e => e).then(x => x.error ? x.stderr : x.stdout);
    
    if (output.indexOf("Successfully deleted") != -1) {
        res.send("OK");
    } else {
        res.send(output);
    }
};

module.exports = (app) => {
    app.get("/", index_page);
    app.get("/detail/:run_id([1-9][0-9]{0,7})$", detail_page);
    app.get("/detail/:run_id([1-9][0-9]{0,7})/stdout$", stdout_content);
    app.get("/detail/:run_id([1-9][0-9]{0,7})/stderr$", stderr_content);
    app.get("/wishlist", wishlist_page);
    app.post("/wish_add", wish_add);
    app.post("/wish_run", wish_run);
    app.post("/wish_cancel", wish_cancel);
    app.post("/tensorboard_add", tensorboard_add);
    app.post("/tensorboard_del", tensorboard_del);
};
