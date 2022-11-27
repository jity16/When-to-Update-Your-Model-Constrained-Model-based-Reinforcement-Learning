const fs = require("fs");
const fsp = require("fs").promises;
const path = require("path");
const config = require("./config");
const utils = require("./utils");

const query_is_running = async (L, R) => {
    const stdout = await utils.run_process(
        "c/check_running.exe",
        [`${L}`, `${R}`],
        30000,
    ).catch(e => e).then(x => x.error ? "" : x.stdout);
    
    const stdout_lines = stdout.split("\n");
    console.log("is_running:", stdout_lines);
    
    const result = {};
    
    for (const line of stdout_lines) {
        const arr = line.split(" ").map(x => parseInt(x));
        if (arr.length != 2) continue;
        
        result[arr[0]] = arr[1];
    }
    
    for (let i = L; i <= R; i++) {
        if (result[i] == undefined) {
            throw `Internal error: query_is_running(${L}, ${R}) failed`;
        }
    }
    
    return result;
};

const error_run_status = {
    run_id: 'Error',
    env_name: 'Error',
    freq_name: 'Error',
    seed: 'Error',
    gpu_id: 'Error',
    status: 'Error',
    submit_time: 'Error',
    is_linked: false,
    is_running: false,
    is_error: true,
};

const do_query_run = async (id, is_running) => {
    const status_filename = path.join(config.run_status_path, `status_${id}.txt`);
    const status = await fsp.readFile(status_filename, "utf-8")
    .then(x => JSON.parse(x))
    .catch(_ => ({...error_run_status}));
    
    status.is_running = is_running;
    status.is_error = status.is_error || fs.existsSync(path.join(config.run_status_path, `has_error_${id}`));
    status.is_stopped = !is_running && fs.existsSync(path.join(config.run_status_path, `is_manually_stopped_${id}`));
    status.run_id = id;
    status.status = status.is_running ? "Running" : status.is_stopped ? "Stopped" : status.is_error ? "Error" : "Completed";
    
    return status;
};

const query_all = async () => {
    await fsp.mkdir(config.run_status_path, { recursive: true });
    await fsp.mkdir(config.tensorboard_logdir_path, { recursive: true });
    
    const last_run_id_filename = path.join(config.run_status_path, "last_run_id.txt");
    const content = await fsp.readFile(last_run_id_filename, "utf-8").catch(() => null);
    
    let last_run_id = 0;
    if (content !== null) {
        last_run_id = parseInt(content);
        if (!(last_run_id >= 0 && last_run_id <= 100000000)) {
            throw `last_run_id ('${content}') is not a number`;
        }
    }
    
    const is_runnings = await query_is_running(1, last_run_id);
    
    const runs = [null];
    for (let i = 1; i <= last_run_id; i++) {
        runs.push(await do_query_run(i, is_runnings[i]));
    }
    
    return {
        last_run_id,
        runs,
    };
};

const query_run = async (run_id) => {
    const is_running = (await query_is_running(run_id, run_id))[run_id];
    return await do_query_run(run_id, is_running);
};

const read_stdout = async (run_id) => {
    const filename = path.join(config.run_status_path, `stdout_${run_id}.txt`);
    return await fsp.readFile(filename, "utf-8").catch(_ => "");
};

const read_stderr = async (run_id) => {
    const filename = path.join(config.run_status_path, `stderr_${run_id}.txt`);
    return await fsp.readFile(filename, "utf-8").catch(_ => "");
};

module.exports = {
    query_all,
    query_run,
    read_stdout,
    read_stderr,
};
