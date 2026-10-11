// PyTorch .pth（zip 格式 state_dict）浏览器端解析器
// 支持解析 torch.save(model.state_dict(), path) 保存的 .pth 文件（新版 zip 格式）：
// 浏览器内直接读 ZIP 目录 + 用针对 PyTorch 子集的 pickle 虚拟机重建 state_dict，
// 再按权重形状自动推导网络结构，全程无需后端。
//
// 结构约定（与本项目 model.py / improve.py 一致）：
//   - 层命名为 convN / fcN（N 从 1 递增），bnN（可选）紧跟对应 convN
//   - 每个卷积块 = Conv2d(padding=k//2) → [BatchNorm2d] → ReLU → MaxPool2d(2,2)
//   - 全连接层之间用 ReLU（最后一层除外），输出 10 维 logits
window.PTHLoader = (function () {
  'use strict';

  const DTYPES = {
    'FloatStorage':    { size: 4, cast: 'f32' },
    'DoubleStorage':   { size: 8, cast: 'f64' },
    'HalfStorage':     { size: 2, cast: 'f16' },
    'LongStorage':     { size: 8, cast: 'i64' },
    'IntStorage':      { size: 4, cast: 'i32' },
    'ShortStorage':    { size: 2, cast: 'i16' },
    'CharStorage':     { size: 1, cast: 'i8'  },
    'ByteStorage':     { size: 1, cast: 'u8'  },
    'BoolStorage':     { size: 1, cast: 'bool'},
    'BFloat16Storage': { size: 2, cast: 'bf16'},
  };

  // ---------------- ZIP 解析（读中央目录；deflate 用浏览器原生 DecompressionStream） ----------------
  function parseZip(u8) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let eocd = -1;
    for (let i = u8.length - 22; i >= Math.max(0, u8.length - 65558); i--) {
      if (u8[i] === 0x50 && u8[i + 1] === 0x4b && u8[i + 2] === 0x05 && u8[i + 3] === 0x06) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('不是 zip 格式的 .pth 文件（可能是旧版 torch 的 legacy 格式，请用新版 torch 重新保存）');

    const count = dv.getUint16(eocd + 10, true);
    let p = dv.getUint32(eocd + 16, true);
    const entries = new Map();
    const decoder = new TextDecoder();
    for (let i = 0; i < count; i++) {
      if (dv.getUint32(p, true) !== 0x02014b50) break;
      const method = dv.getUint16(p + 10, true);
      const csize = dv.getUint32(p + 20, true);
      const nlen = dv.getUint16(p + 28, true);
      const elen = dv.getUint16(p + 30, true);
      const clen = dv.getUint16(p + 32, true);
      const lho = dv.getUint32(p + 42, true);
      const name = decoder.decode(u8.subarray(p + 46, p + 46 + nlen));
      entries.set(name, { method, csize, lho });
      p += 46 + nlen + elen + clen;
    }

    async function read(name) {
      const e = entries.get(name);
      if (!e) throw new Error('压缩包内缺少文件 ' + name);
      const nlen = dv.getUint16(e.lho + 26, true);
      const elen = dv.getUint16(e.lho + 28, true);
      const start = e.lho + 30 + nlen + elen;
      const raw = u8.subarray(start, start + e.csize);
      if (e.method === 0) return raw;
      if (e.method === 8) return await inflateRaw(raw);
      throw new Error('不支持的 zip 压缩方式 ' + e.method);
    }
    return { read, names: () => [...entries.keys()] };
  }

  async function inflateRaw(u8) {
    const ds = new DecompressionStream('deflate-raw');
    const stream = new Blob([u8]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  // ---------------- pickle 虚拟机（覆盖 torch state_dict 用到的操作码子集） ----------------
  const MARK = Symbol('pickle-mark');

  function parsePickle(u8, persistentLoad) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let pos = 0;
    const stack = [];
    const memo = [];

    const take = (n) => { if (pos + n > u8.length) throw new Error('pickle 数据不完整'); const v = pos; pos += n; return v; };
    const u1 = () => u8[take(1)];
    const u2 = () => dv.getUint16(take(2), true);
    const i4 = () => dv.getInt32(take(4), true);
    const u4 = () => dv.getUint32(take(4), true);
    const u8le = () => { const a = dv.getUint32(take(4), true), b = dv.getUint32(take(4), true); return b * 4294967296 + a; };
    const f8be = () => dv.getFloat64(take(8), false);   // BINFLOAT 是大端
    const latinLine = () => { let s = ''; while (pos < u8.length && u8[pos] !== 0x0a) s += String.fromCharCode(u8[pos++]); pos++; return s; };
    const utf8 = (n) => { const s = take(n); return new TextDecoder().decode(u8.subarray(s, s + n)); };

    function popMarked() {
      let i = stack.length - 1;
      while (i >= 0 && stack[i] !== MARK) i--;
      if (i < 0) throw new Error('pickle MARK 不匹配');
      const items = stack.slice(i + 1);
      stack.length = i;
      return items;
    }

    for (;;) {
      if (pos >= u8.length) throw new Error('pickle 缺少 STOP 结束符');
      const op = u8[pos++];
      switch (op) {
        case 0x80: take(1); break;                                   // PROTO
        case 0x95: take(8); break;                                   // FRAME（整体读入，忽略）
        case 0x5d: stack.push([]); break;                            // EMPTY_LIST
        case 0x7d: stack.push({}); break;                            // EMPTY_DICT
        case 0x29: stack.push([]); break;                            // EMPTY_TUPLE
        case 0x4e: stack.push(null); break;                          // NONE
        case 0x88: stack.push(true); break;                          // NEWTRUE（协议 2）
        case 0x89: stack.push(false); break;                         // NEWFALSE（协议 2）
        case 0x46: stack.push(parseFloat(latinLine())); break;       // FLOAT（文本协议）
        case 0x4a: stack.push(i4()); break;                          // BININT
        case 0x4b: stack.push(u1()); break;                          // BININT1
        case 0x4d: stack.push(u2()); break;                          // BININT2
        case 0x8a: {                                                 // LONG1
          const n = u1(); const s = take(n); let v = 0n;
          for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(u8[s + i]);
          stack.push(Number(v)); break;
        }
        case 0x47: stack.push(f8be()); break;                        // BINFLOAT
        case 0x8c: stack.push(utf8(u1())); break;                    // SHORT_BINUNICODE
        case 0x58: stack.push(utf8(u4())); break;                    // BINUNICODE
        case 0x8d: stack.push(utf8(u8le())); break;                  // BINUNICODE8
        case 0x43: { const n = u1(); const s = take(n); stack.push(u8.slice(s, s + n)); break; } // SHORT_BINBYTES
        case 0x42: { const n = u4(); const s = take(n); stack.push(u8.slice(s, s + n)); break; } // BINBYTES
        case 0x28: stack.push(MARK); break;                          // MARK
        case 0x74: stack.push(popMarked()); break;                   // TUPLE
        case 0x85: { const a = stack.pop(); stack.push([a]); break; }                            // TUPLE1
        case 0x86: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; }        // TUPLE2
        case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
        case 0x6c: stack.push(popMarked()); break;                   // LIST
        case 0x61: { const v = stack.pop(); stack[stack.length - 1].push(v); break; }            // APPEND
        case 0x65: { const items = popMarked(); stack[stack.length - 1].push(...items); break; } // APPENDS
        case 0x64: {                                                 // DICT
          const items = popMarked(); const obj = {};
          for (let i = 0; i + 1 < items.length; i += 2) obj[items[i]] = items[i + 1];
          stack.push(obj); break;
        }
        case 0x73: { const v = stack.pop(), k = stack.pop(); stack[stack.length - 1][k] = v; break; }       // SETITEM
        case 0x75: {                                                 // SETITEMS
          const items = popMarked(); const obj = stack[stack.length - 1];
          for (let i = 0; i + 1 < items.length; i += 2) obj[items[i]] = items[i + 1];
          break;
        }
        case 0x94: memo.push(stack[stack.length - 1]); break;        // MEMOIZE
        case 0x71: memo[u1()] = stack[stack.length - 1]; break;      // BINPUT
        case 0x72: memo[u4()] = stack[stack.length - 1]; break;      // LONG_BINPUT
        case 0x67: stack.push(memo[parseInt(latinLine(), 10)]); break; // GET
        case 0x68: stack.push(memo[u1()]); break;                    // BINGET
        case 0x6a: stack.push(memo[u4()]); break;                    // LONG_BINGET
        case 0x63: { const m = latinLine(); const n = latinLine(); stack.push({ __global__: m + '.' + n }); break; } // GLOBAL
        case 0x93: { const n = stack.pop(), m = stack.pop(); stack.push({ __global__: m + '.' + n }); break; }       // STACK_GLOBAL
        case 0x52: {                                                 // REDUCE
          const args = stack.pop(), f = stack.pop();
          stack.push(reduceGlobal(f, args));
          break;
        }
        case 0x81: { const args = stack.pop(), cls = stack.pop(); stack.push({ __unknown__: cls && cls.__global__ }); break; } // NEWOBJ
        case 0x62: {                                                 // BUILD
          const state = stack.pop(); const obj = stack[stack.length - 1];
          if (obj && typeof obj === 'object' && !Array.isArray(obj) && state && typeof state === 'object' && !Array.isArray(state)) {
            Object.assign(obj, state);
          }
          break;
        }
        case 0x51: stack.push(persistentLoad(stack.pop())); break;   // BINPERSID
        case 0x30: stack.pop(); break;                               // POP
        case 0x31: popMarked(); break;                               // POP_MARK
        case 0x2e: return stack[0];                                  // STOP
        default: throw new Error('暂不支持的 pickle 操作码 0x' + op.toString(16) + '（文件可能不是 state_dict）');
      }
    }

    function reduceGlobal(f, args) {
      const name = f && f.__global__;
      if (name === 'torch._utils._rebuild_tensor_v2' || name === 'torch._utils._rebuild_tensor_v3') {
        const [storage, offset, size, stride] = args;
        return { __tensor__: storage, offset: offset || 0, size, stride };
      }
      if (name === 'collections.OrderedDict') {
        const obj = {};
        for (const [k, v] of (args[0] || [])) obj[k] = v;
        return obj;
      }
      return { __unknown__: name || String(f) };
    }
  }

  function persistentLoad(pid) {
    if (Array.isArray(pid) && pid[0] === 'storage') {
      const typeGlobal = pid[1] && pid[1].__global__;
      const suffix = typeGlobal ? typeGlobal.split('.').pop() : '';
      const dt = DTYPES[suffix];
      if (!dt) throw new Error('暂不支持的存储类型 ' + typeGlobal);
      return { __storage__: { key: String(pid[2]), cast: dt.cast, elemsize: dt.size, numel: pid[4] } };
    }
    throw new Error('不支持的持久化对象（可能不是 state_dict）');
  }

  // ---------------- 张量数据读取（按 stride 展开为连续 Float32Array） ----------------
  function f16ToF32(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, m = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  function bf16ToF32(h) {
    const buf = new ArrayBuffer(4), v = new Uint32Array(buf);
    v[0] = h << 16;
    return new Float32Array(buf)[0];
  }

  async function tensorToFloat32(v, zip, dataFiles) {
    const st = v.__tensor__.__storage__;
    const name = dataFiles.get(st.key);
    if (!name) throw new Error('压缩包内缺少张量数据文件 data/' + st.key);
    const bytes = await zip.read(name);
    const size = v.size, nd = size.length;
    const numel = size.reduce((a, b) => a * b, 1);
    const out = new Float32Array(numel);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    // 连续性判断（C 顺序 stride），连续时走快路径
    let contiguous = true, acc = 1;
    for (let d = nd - 1; d >= 0; d--) { if (v.stride[d] !== acc) { contiguous = false; break; } acc *= size[d]; }

    for (let i = 0; i < numel; i++) {
      let idx = i;
      if (!contiguous) {
        idx = 0; let rem = i;
        for (let d = nd - 1; d >= 0; d--) { const c = rem % size[d]; rem = (rem - c) / size[d]; idx += c * v.stride[d]; }
        idx += v.offset;
      } else {
        idx = i + v.offset;
      }
      const off = idx * st.elemsize;
      switch (st.cast) {
        case 'f32': out[i] = dv.getFloat32(off, true); break;
        case 'f64': out[i] = dv.getFloat64(off, true); break;
        case 'f16': out[i] = f16ToF32(dv.getUint16(off, true)); break;
        case 'bf16': out[i] = bf16ToF32(dv.getUint16(off, true)); break;
        case 'u8': out[i] = dv.getUint8(off); break;
        case 'i8': out[i] = dv.getInt8(off); break;
        case 'bool': out[i] = dv.getUint8(off) ? 1 : 0; break;
        case 'i16': out[i] = dv.getInt16(off, true); break;
        case 'i32': out[i] = dv.getInt32(off, true); break;
        case 'i64': out[i] = Number(dv.getBigInt64(off, true)); break;
        default: throw new Error('不支持的数值类型 ' + st.cast);
      }
    }
    return { size: size.slice(), data: out };
  }

  // ---------------- 主入口：解析 .pth → { key: {size, data} } ----------------
  async function parsePth(buffer) {
    const u8 = new Uint8Array(buffer);
    if (!(u8[0] === 0x50 && u8[1] === 0x4b)) {
      throw new Error('不是 zip 格式的 .pth（旧版 torch 的 legacy 格式暂不支持，请用新版 torch 重新保存）');
    }
    const zip = parseZip(u8);

    let pklName = null;
    const dataFiles = new Map();
    for (const name of zip.names()) {
      if (name.endsWith('/data.pkl') || name === 'data.pkl') pklName = name;
      const m = name.match(/\/data\/([^/]+)$/);
      if (m) dataFiles.set(m[1], name);
    }
    if (!pklName) throw new Error('压缩包内没有 data.pkl，可能不是 state_dict 文件');

    const pklBytes = await zip.read(pklName);
    const raw = parsePickle(pklBytes, persistentLoad);
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new Error('这不是 state_dict（请用 torch.save(model.state_dict(), path) 保存）');
    }

    const state = {};
    for (const [k, v] of Object.entries(raw)) {
      if (v && v.__tensor__ && v.__tensor__.__storage__) {
        state[k] = await tensorToFloat32(v, zip, dataFiles);
      }
    }
    if (Object.keys(state).length === 0) throw new Error('state_dict 中没有可识别的张量');
    return state;
  }

  // ---------------- 结构推导：convN/fcN 命名 → 可执行的前向描述 ----------------
  function nest4(data, oc, ic, kh, kw) {
    const w = [];
    let p = 0;
    for (let o = 0; o < oc; o++) {
      const chan = [];
      for (let c = 0; c < ic; c++) {
        const rows = [];
        for (let y = 0; y < kh; y++) { rows.push(data.slice(p, p + kw)); p += kw; }
        chan.push(rows);
      }
      w.push(chan);
    }
    return w;
  }
  function nest2(data, rows, cols) {
    const w = [];
    let p = 0;
    for (let i = 0; i < rows; i++) { w.push(data.slice(p, p + cols)); p += cols; }
    return w;
  }

  function buildModelFromStateDict(state) {
    const convRe = /^(?:module\.)?conv(\d+)\.(weight|bias)$/;
    const bnRe = /^(?:module\.)?bn(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$/;
    const fcRe = /^(?:module\.)?fc(\d+)\.(weight|bias)$/;
    const convs = {}, bns = {}, fcs = {};
    for (const [k, v] of Object.entries(state)) {
      let m = k.match(convRe);
      if (m) { (convs[m[1]] = convs[m[1]] || {})[m[2]] = v; continue; }
      m = k.match(bnRe);
      if (m) { (bns[m[1]] = bns[m[1]] || {})[m[2]] = v; continue; }
      m = k.match(fcRe);
      if (m) { (fcs[m[1]] = fcs[m[1]] || {})[m[2]] = v; }
    }
    const convIdx = Object.keys(convs).map(Number).sort((a, b) => a - b);
    const fcIdx = Object.keys(fcs).map(Number).sort((a, b) => a - b);
    if (!fcIdx.length) throw new Error('未找到 fcN 命名的全连接层（当前支持 convN/fcN 命名的 state_dict）');

    let C = 1, H = 28, W = 28;
    const convDesc = [], convLayers = [];
    for (const n of convIdx) {
      const c = convs[n];
      if (!c.weight || !c.bias) throw new Error('conv' + n + ' 缺少 weight 或 bias');
      const [oc, ic, kh, kw] = c.weight.size;
      if (kw !== kh) throw new Error('conv' + n + ' 仅支持方形卷积核');
      if (ic !== C) throw new Error(`conv${n} 的输入通道为 ${ic}，与上一层输出通道 ${C} 不衔接`);
      if (n === convIdx[0] && ic !== 1) throw new Error('第一层卷积的输入通道必须为 1（28×28 灰度图），当前为 ' + ic);
      const layer = { w: nest4(c.weight.data, oc, ic, kh, kw), b: c.bias.data, k: kh, pad: kh >> 1 };
      let bnTag = '';
      if (bns[n]) {
        const bn = bns[n];
        for (const f of ['weight', 'bias', 'running_mean', 'running_var']) {
          if (!bn[f]) throw new Error(`bn${n} 缺少 ${f}（BatchNorm 参数不完整，请确认保存的是 model.state_dict()）`);
        }
        if (bn.weight.size[0] !== oc) throw new Error(`bn${n} 的通道数 ${bn.weight.size[0]} 与 conv${n} 输出通道 ${oc} 不一致`);
        layer.bn = { gamma: bn.weight.data, beta: bn.bias.data, mean: bn.running_mean.data, var: bn.running_var.data };
        bnTag = '+BN';
      }
      convLayers.push(layer);
      convDesc.push(`conv${n}: ${ic}→${oc}(k${kh})${bnTag}`);
      C = oc; H = Math.floor(H / 2); W = Math.floor(W / 2);
    }

    const flatDim = convIdx.length ? C * H * W : 784;
    const fcDesc = [];
    const fcLayers = [];
    let prev = flatDim;
    for (const n of fcIdx) {
      const f = fcs[n];
      if (!f.weight || !f.bias) throw new Error('fc' + n + ' 缺少 weight 或 bias');
      const [outF, inF] = f.weight.size;
      if (inF !== prev) {
        throw new Error(`fc${n} 的输入维度为 ${inF}，与上一层输出 ${prev} 不衔接（按本页约定结构推导的展平维度为 ${flatDim}）`);
      }
      if (f.bias.size[0] !== outF) throw new Error('fc' + n + ' 的 bias 形状与 weight 不匹配');
      fcLayers.push({ w: nest2(f.weight.data, outF, inF), b: f.bias.data });
      fcDesc.push(`${inF}→${outF}`);
      prev = outF;
    }
    if (prev !== 10) throw new Error(`输出层维度为 ${prev}，应为 10（数字 0~9 共 10 类）`);

    const kind = convIdx.length ? 'CNN' : 'MLP';
    const desc = `${kind} · ${convDesc.length ? convDesc.join(' → ') + ' → ' : ''}fc ${fcDesc.join(' → ')}`;
    return { convs: convLayers, fcs: fcLayers, desc };
  }

  return { parsePth, buildModelFromStateDict };
})();
