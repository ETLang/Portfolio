// Read-only structural inspector for a scene JSON (public/scenes/*.json). Exists to replace ad hoc
// `node -e`/`python -c` one-liners that were being re-typed per query and re-approved every time
// (see MEMORY / CLAUDE.md discussion) - this script's args are all plain strings used as property
// lookups/filters, never eval'd, so the exact same command line is safe to allowlist regardless of
// which scene/id/name is being asked about.
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const scenesDir = path.join(__dirname, '..', '..', '..', 'public', 'scenes');

function parseArgs(argv) {
    const args = { mode: 'summary' };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--scene') args.scene = argv[++i];
        else if (a === '--keys') args.mode = 'keys';
        else if (a === '--summary') args.mode = 'summary';
        else if (a === '--section') { args.mode = 'section'; args.key = argv[++i]; }
        else if (a === '--names') args.mode = 'names';
        else if (a === '--array') args.array = argv[++i];
        else if (a === '--id') { args.mode = 'id'; args.id = argv[++i]; }
        else if (a === '--tree') args.mode = 'tree';
        else if (a === '--root') args.root = argv[++i];
        else if (a === '--find') { args.mode = 'find'; args.query = argv[++i]; }
        else throw new Error(`Unknown argument: ${a}`);
    }
    if (!args.scene) throw new Error('--scene is required');
    return args;
}

function resolveScenePath(scene) {
    if (existsSync(scene)) return scene;
    const candidate = path.join(scenesDir, scene);
    if (existsSync(candidate)) return candidate;
    const withExt = candidate.endsWith('.json') ? candidate : `${candidate}.json`;
    if (existsSync(withExt)) return withExt;
    throw new Error(`Scene file not found: ${scene}`);
}

function describeValue(v) {
    if (Array.isArray(v)) return `array(${v.length})`;
    if (v && typeof v === 'object') return `object{${Object.keys(v).join(',')}}`;
    return JSON.stringify(v);
}

function collectArrays(data) {
    return Object.entries(data).filter(([, v]) => Array.isArray(v));
}

function cmdKeys(data) {
    for (const k of Object.keys(data)) console.log(`${k}: ${describeValue(data[k])}`);
}

function cmdSummary(data) {
    for (const [k, v] of Object.entries(data)) {
        if (Array.isArray(v)) {
            const itemKeys = v.length ? Object.keys(v[0]).join(', ') : '';
            console.log(`${k}: array(${v.length})${itemKeys ? ' - item keys: ' + itemKeys : ''}`);
        } else {
            console.log(`${k}: ${describeValue(v)}`);
        }
    }
}

function cmdSection(data, key) {
    if (!(key in data)) throw new Error(`No top-level key "${key}". Available: ${Object.keys(data).join(', ')}`);
    console.log(JSON.stringify(data[key], null, 2));
}

function cmdNames(data, arrayKey) {
    const key = arrayKey || 'objects';
    const arr = data[key];
    if (!Array.isArray(arr)) throw new Error(`"${key}" is not an array. Available: ${Object.keys(data).join(', ')}`);
    const counts = new Map();
    for (const item of arr) {
        const name = item && item.name !== undefined ? String(item.name) : '(no name)';
        counts.set(name, (counts.get(name) || 0) + 1);
    }
    const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    for (const [name, count] of sorted) console.log(`${count}\t${name}`);
}

function cmdId(data, id) {
    const idNum = Number(id);
    const target = Number.isNaN(idNum) ? id : idNum;
    let found = 0;
    for (const [arrayKey, arr] of collectArrays(data)) {
        for (const item of arr) {
            if (!item || typeof item !== 'object') continue;
            const matchedFields = ['id', 'objectId', 'ownerId', 'parentId'].filter(
                (field) => field in item && item[field] === target
            );
            if (matchedFields.length) {
                found++;
                console.log(`[${arrayKey}] matched via ${matchedFields.join(', ')}:`);
                console.log(JSON.stringify(item, null, 2));
                console.log();
            }
        }
    }
    if (!found) console.log(`No entries reference id ${id} in any top-level array.`);
}

function cmdTree(data, rootId) {
    const objs = data.objects;
    if (!Array.isArray(objs)) throw new Error('No "objects" array in this scene.');
    const ROOT = 'ROOT'; // scenes use -1 (and sometimes absent) as the "no parent" sentinel, not null
    const normalizeParent = (o) => (o.parentId === undefined || o.parentId === -1 ? ROOT : o.parentId);
    const childrenOf = new Map();
    for (const o of objs) {
        const p = normalizeParent(o);
        if (!childrenOf.has(p)) childrenOf.set(p, []);
        childrenOf.get(p).push(o);
    }
    function print(o, depth) {
        console.log(`${'  '.repeat(depth)}${o.id}\t${o.name}${o.active === false ? ' (inactive)' : ''}`);
        for (const child of childrenOf.get(o.id) || []) print(child, depth + 1);
    }
    if (rootId !== undefined) {
        const rootNum = Number(rootId);
        const root = objs.find((o) => o.id === rootNum);
        if (!root) throw new Error(`No object with id ${rootId}`);
        print(root, 0);
    } else {
        for (const root of childrenOf.get(ROOT) || []) print(root, 0);
    }
}

function cmdFind(data, query, arrayKey) {
    const q = query.toLowerCase();
    const arraysToSearch = arrayKey ? [[arrayKey, data[arrayKey]]] : collectArrays(data);
    let found = 0;
    for (const [key, arr] of arraysToSearch) {
        if (!Array.isArray(arr)) continue;
        for (const item of arr) {
            if (item && typeof item.name === 'string' && item.name.toLowerCase().includes(q)) {
                found++;
                console.log(`[${key}] ${JSON.stringify(item)}`);
            }
        }
    }
    if (!found) console.log(`No entries with name matching "${query}".`);
}

function main() {
    const args = parseArgs(process.argv.slice(2));
    const data = JSON.parse(readFileSync(resolveScenePath(args.scene), 'utf8'));
    switch (args.mode) {
        case 'keys': return cmdKeys(data);
        case 'summary': return cmdSummary(data);
        case 'section': return cmdSection(data, args.key);
        case 'names': return cmdNames(data, args.array);
        case 'id': return cmdId(data, args.id);
        case 'tree': return cmdTree(data, args.root);
        case 'find': return cmdFind(data, args.query, args.array);
    }
}

try {
    main();
} catch (err) {
    console.error(String((err && err.message) || err));
    process.exitCode = 1;
}
