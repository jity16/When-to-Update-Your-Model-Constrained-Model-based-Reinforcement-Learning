const fs = require("fs");
const path = require("path");
const config = require("./config");
const utils = require("./utils");

const wishlist_filename = path.join(config.run_status_path, "wishlist.txt");

const empty_wishlist = [ null ];

const init_wishlist = () => {
    fs.writeFileSync(wishlist_filename, JSON.stringify(empty_wishlist), "utf-8");
};

const query_all = async () => {
    if (!fs.existsSync(wishlist_filename)) {
        init_wishlist();
    }
    
    try {
        const wishlist_content = fs.readFileSync(wishlist_filename, "utf-8");
        const wishlist = JSON.parse(wishlist_content);
        return wishlist;
    } catch (_) {
        throw "Can not read wishlist";
    }
};

const write_wishlist = async (wishlist) => {
    const content = JSON.stringify(wishlist);
    fs.writeFileSync(wishlist_filename + ".tmp", content, "utf-8");
    fs.renameSync(wishlist_filename + ".tmp", wishlist_filename);
};

// IN: data { algo, env, freq, seed, time }
// OUT: id
const add = async (data) => {
    const wishlist = await query_all();
    wishlist.push({
        id: wishlist.length,
        ...data,
    });
    
    await write_wishlist(wishlist);
    
    return wishlist.length - 1;
};

// IN: id, data { <fields_to_update> }, prereqs { <fields_to_assert> }
// OUT: updated wishlist[id]
const update = async (id, data, prereqs) => {
    const wishlist = await query_all();
    
    if (!(id > 0 && id < wishlist.length)) {
        throw "Invalid id";
    }
    
    if (prereqs) for (const key in prereqs) {
        if (wishlist[id][key] != prereqs[key]) {
            throw `Expected ${key} == '${prereqs[key]}', got '${wishlist[id][key]}'`;
        }
    }
    
    for (const key in data) {
        wishlist[id][key] = data[key];
    }
    
    await write_wishlist(wishlist);
    
    return wishlist[id];
};

const find_best_gpu = async () => {
    const stdout = await utils.run_process(
        "gpustat",
        [],
        30000,
    ).catch(e => e).then(x => x.error ? "" : x.stdout);
    
    let best_gpu_id = -1;
    let min_mem_usage = 1000000000;
    
    for (const line of stdout.split("\n")) {
        if (line.indexOf("MB") == -1) continue;
        const gpu_id = parseInt(line.substr(1).split("]")[0]);
        const mem_usage = parseInt(line.split("|")[2].split("/")[0].trim());
        if (mem_usage < min_mem_usage) {
            min_mem_usage = mem_usage;
            best_gpu_id = gpu_id;
        }
    }
    
    return best_gpu_id;
};

// IN: id
const run = async (id) => {
    const wishlist = await query_all();
    
    if (!(id > 0 && id < wishlist.length)) {
        throw "Invalid id";
    }
    
    if (wishlist[id].status != "Wishing") {
        throw `Wish ${id}'s status is not 'Wishing'`;
    }
    
    const wish = wishlist[id];
    const algo_name = wish.algo_name;
    const env_name = wish.env_name;
    const freq_name = wish.freq_name;
    const seed = wish.seed;
    const gpu_id = await find_best_gpu();
    
    const stdout = await utils.run_process(
        "bash",
        ["-c", `cd .. && tools/run ${algo_name} ${env_name} ${freq_name} ${seed} ${gpu_id}`],
        30000,
    ).catch(e => e).then(x => x.error ? x.stderr : x.stdout);
    
    if (stdout.indexOf("started") == -1) {
        throw `tools/run returned '${stdout}'`
    }
};

module.exports = {
    query_all,
    add,
    update,
    run,
};
