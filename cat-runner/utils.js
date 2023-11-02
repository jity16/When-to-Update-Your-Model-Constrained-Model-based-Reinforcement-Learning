// Utils

const utils = module.exports = {};
const sprintf = utils.sprintf = require("sprintf-js").sprintf;
const child_process = require("child_process");

// Timestamp to (formatted) datetime
utils.timestamp_to_datetime = (timestamp) => {
    const date = new Date(timestamp);
    
    const year = date.getFullYear();
    const day = date.getDate();
    const month = (date.getMonth() + 1);
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const seconds = date.getSeconds();
    
    return sprintf("%d-%02d-%02d %02d:%02d:%02d",
        year, month, day, hours, minutes, seconds);
};

// Current datetime
utils.current_datetime = () => {
    return utils.timestamp_to_datetime(Date.now());
};

// Current timestamp
utils.current_timestamp = () => {
    return Date.now();
};

// ms to string (for countdown)
utils.ms_to_countdown_string = (ms) => {
    const s = Math.floor(ms / 1000) + 1;
    const hours = Math.floor(s / 3600);
    const minutes = Math.floor(s % 3600 / 60);
    const seconds = s % 60;
    return `${hours} 小时 ${minutes} 分钟 ${seconds} 秒`;
};

// Wait ms
utils.wait = (ms) => {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
};

// Simple queue
utils.simple_queue = () => {
    let left = [];
    let right = [];
    
    const ret = {
        length: 0,
        front: null,
        back: null,
        push_back: null,
        pop_front: null,
    };
    
    ret.front = () => {
        if (left.length == 0) {
            return right[0];
        } else {
            return left[left.length - 1];
        }
    };
    
    ret.back = () => {
        if (right.length == 0) {
            return left[0];
        } else {
            return right[right.length - 1];
        }
    };
    
    ret.push_back = (x) => {
        return right.push(x), ++ret.length;
    };
    
    ret.pop_front = () => {
        if (ret.length == 0) return undefined;
        
        if (left.length == 0) {
            left = right.reverse();
            right = [];
        }
        
        const res = (ret.length--, left.pop());
        return res;
    };
    
    return ret;
};

// Simple async mutex lock
utils.lock = () => {
    let locked = false;
    const requests = utils.simple_queue();
    
    const acquire = async () => {
        if (!locked) {
            locked = true;
        } else {
            const req = { resolve: null };
            const promise = new Promise((r) => { req.resolve = r });
            requests.push_back(req);
            await promise;
        }
    };
    
    const release = async () => {
        if (requests.length) {
            requests.pop_front().resolve();
        } else {
            locked = false;
        }
    };
    
    return {
        acquire: acquire,
        release: release,
    };
};

utils.run_process = (name, args, ms, maxBuffer = 1024 * 100) => {
    return new Promise((resolve, reject) => {
        child_process.execFile(name, args, {
            timeout: ms,
            maxBuffer: maxBuffer,
        }, (error, stdout, stderr) => {
            (error ? reject : resolve)({
                error: error,
                stdout: stdout,
                stderr: stderr,
            });
        });
    });
};

utils.and_more = (s, max_len = 10240) => {
    if (s.length > max_len) {
        return s.substr(0, max_len) + "\n\n(... and more)";
    } else {
        return s;
    }
};

utils.and_more_middle = (s, max_len = 10240) => {
    if (s.length > max_len) {
        const n_lines = s.substr(Math.floor(max_len / 2), s.length - max_len).split("\n").length;
        return s.substr(0, Math.floor(max_len / 2))
            + `\n\n (... ${n_lines} more lines ...)\n\n`
            + s.substr(s.length - max_len + Math.floor(max_len / 2));
    } else {
        return s;
    }
};
