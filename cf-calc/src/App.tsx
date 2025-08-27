import React, { useMemo, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

// ------------------------------------------------------------
// Continued Fraction Calculator (single-file React component)
// Features:
// - Parse [a0; a1, a2, ...] form (finite) and periodic (infinite)
// - Periodic notations supported:
//   1) [a0; prefix, (b1, b2, ...)]  // (...) is the repeating block
//   2) [a0; prefix | b1, b2, ...]   // right side of '|' is the repeating block
//   3) [a0; 1,2,1,2,...]            // trailing ... auto-detects the last repeating block
// - Exact value shown as LaTeX: rational p/q or quadratic surd (P + Q√Δ)/R
// - Approximation value and convergents table
// - Precise integer arithmetic with BigInt
// - Modern UI (Tailwind)
// ------------------------------------------------------------

// ---------- Utilities (BigInt) ----------
const BI = (n: number | string | bigint) => BigInt(n);
const absBI = (x: bigint) => (x < 0n ? -x : x);
const gcdBI = (a: bigint, b: bigint): bigint => {
  a = absBI(a); b = absBI(b);
  while (b !== 0n) { const t = a % b; a = b; b = t; }
  return a;
};


// Matrix [[a,b],[c,d]] × [[e,f],[g,h]]
const matMul = (M: bigint[][], N: bigint[][]): bigint[][] => {
  const [[a,b],[c,d]] = M; const [[e,f],[g,h]] = N;
  return [
    [a*e + b*g, a*f + b*h],
    [c*e + d*g, c*f + d*h]
  ];
};
// T_a = [[a,1],[1,0]]
const T = (a: bigint) => [[a,1n],[1n,0n]] as bigint[][];

// Matrix product from a finite sequence of partial quotients
const matFromSeq = (seq: bigint[]): bigint[][] => {
  let M = [[1n,0n],[0n,1n]] as bigint[][]; // identity
  for (const a of seq) M = matMul(M, T(a));
  return M;
};

// Convergents p_k/q_k from a sequence
const convergents = (seq: bigint[]): {p: bigint, q: bigint}[] => {
  // For M = [[p_k, p_{k-1}],[q_k, q_{k-1}]]
  const res: {p: bigint, q: bigint}[] = [];
  let M = [[1n,0n],[0n,1n]] as bigint[][];
  for (const a of seq) {
    M = matMul(M, T(a));
    res.push({ p: M[0][0], q: M[1][0] });
  }
  return res;
};

// ---------- Parsing ----------
type ParsedCF = {
  a0: bigint;           // first term
  prefix: bigint[];     // a1..ak (before the period)
  period: bigint[];     // b1..bm (repeating block)
  isPeriodic: boolean;  // is periodic
  finiteAll: bigint[];  // whole finite sequence for display
};

const parseCF = (raw: string): ParsedCF => {
  const s = raw.trim().replace(/\s+/g, "");
  if (!s) throw new Error("Input is empty");
  const m = s.match(/^\[?([^\[\];]+);(.+?)\]?$/);
  if (!m) {
    // rescue if brackets are omitted
    const m2 = s.match(/^([^;]+);(.+)$/);
    if (!m2) throw new Error("Please use the form [a0; a1, a2, ...]");
    return parseCF(`[${m2[1]};${m2[2]}]`);
  }
  const a0 = BI(m[1]);
  let rest = m[2];

  // 1) Parentheses for period
  const paren = rest.match(/^(.*)\(([^()]*)\)\s*$/);
  if (paren) {
    const prefixStr = paren[1].replace(/^,|,$/g, "");
    const periodStr = paren[2];
    const prefix = prefixStr ? prefixStr.split(",").filter(Boolean).map(BI) : [];
    const period = periodStr ? periodStr.split(",").filter(Boolean).map(BI) : [];
    if (period.length === 0) throw new Error("The repeating block inside () is empty");
    return {
      a0, prefix, period, isPeriodic: true,
      finiteAll: [a0, ...prefix]
    };
  }

  // 2) '|' for period
  if (rest.includes("|")) {
    const [left, right] = rest.split("|");
    const prefix = left.replace(/^,|,$/g, "").split(",").filter(Boolean).map(BI);
    const period = right.replace(/^,|,$/g, "").split(",").filter(Boolean).map(BI);
    if (period.length === 0) throw new Error("The right side of '|' (period) is empty");
    return {
      a0, prefix, period, isPeriodic: true,
      finiteAll: [a0, ...prefix]
    };
  }

  // 3) trailing ... for automatic period detection
  if (rest.endsWith("...")) {
    const listStr = rest.slice(0, -3).replace(/,\s*$/, "");
    const arr = listStr.split(",").filter(Boolean).map(BI);
    if (arr.length === 0) throw new Error("You need at least one term before ...");
    // guess the shortest repeating block from the tail
    let period: bigint[] | null = null;
    for (let L = 1; L <= Math.floor(arr.length / 2); L++) {
      const tail = arr.slice(-2*L);
      const a = tail.slice(0, L).join(",");
      const b = tail.slice(L).join(",");
      if (a === b) { period = tail.slice(0, L); break; }
    }
    if (!period) {
      // e.g., [1; 2,3,...] => repeat the last element
      period = [arr[arr.length - 1]];
    }
    const prefix = arr.slice(0, arr.length - period.length);
    return {
      a0, prefix, period, isPeriodic: true,
      finiteAll: [a0, ...arr]
    };
  }

  // 4) finite sequence
  const arr = rest.split(",").filter(Boolean).map(BI);
  return { a0, prefix: arr, period: [], isPeriodic: false, finiteAll: [a0, ...arr] };
};

// ---------- Finite evaluation ----------
const evalFiniteExact = (a0: bigint, rest: bigint[]) => {
  const seq = [a0, ...rest];
  const conv = convergents(seq);
  const { p, q } = conv[conv.length - 1];
  return { p, q };
};

// ---------- Periodic → quadratic surd ----------
type Surd = { P: bigint; Q: bigint; R: bigint; Delta: bigint; approx: number; rootSign: 1 | -1 };

const surdFromPeriodic = (prefix: bigint[], period: bigint[]): Surd => {
  const Mp = matFromSeq(prefix); // [[α,β],[γ,δ]]
  const [alpha, beta] = Mp[0];
  const [gamma, delta] = Mp[1];

  const Mr = matFromSeq(period); // [[A,B],[C,D]]
  const [A, B] = Mr[0];
  const [C, D] = Mr[1];

  if (C === 0n) {
    // Degenerate case
    const yP = B; const yR = (D - A);
    const num = alpha * yP + beta * yR;
    const den = gamma * yP + delta * yR;
    const g = gcdBI(num, den);
    let P = num / g; let R = den / g; let Q = 0n; const Delta = 0n;
    if (R < 0n) { P = -P; R = -R; }
    const approx = Number(P) / Number(R);
    return { P, Q, R, Delta, approx, rootSign: 1 };
  }

  const Delta = (D - A) * (D - A) + 4n * B * C;
  const U = (A - D);
  const W = 2n * C;

  // choose sign via numeric comparison
  const sqrtDelta = Math.sqrt(Number(Delta));
  const y1 = (Number(U) + sqrtDelta) / Number(W);
  const y2 = (Number(U) - sqrtDelta) / Number(W);
  const chooseSign: 1 | -1 = y1 > y2 ? 1 : -1;

  const A1 = alpha * U + beta * W;
  const B1 = alpha * BI(chooseSign);
  const C1 = gamma * U + delta * W;
  const D1 = gamma * BI(chooseSign);

  const P = A1 * C1 - B1 * D1 * Delta;
  const Q = B1 * C1 - A1 * D1;
  let R = C1*C1 - D1*D1*Delta;

  let g = gcdBI(gcdBI(absBI(P), absBI(Q)), absBI(R));
  let Pn = P / g; let Qn = Q / g; let Rn = R / g;
  if (Rn < 0n) { Pn = -Pn; Qn = -Qn; Rn = -Rn; }

  const approx = (Number(Pn) + Number(Qn) * Math.sqrt(Number(Delta))) / Number(Rn);
  return { P: Pn, Q: Qn, R: Rn, Delta, approx, rootSign: chooseSign };
};

// Minimal polynomial of x
const minimalPolynomial = (
  alpha: bigint, beta: bigint, gamma: bigint, delta: bigint, A: bigint, B: bigint, C: bigint, D: bigint
) => {
  const b0 = (D - A);
  // y = (δ x - β) / (α - γ x)
  // C(δx - β)^2 + b0(δx - β)(α - γx) - B(α - γx)^2 = 0

  const x2 = C*(delta*delta) - b0*delta*gamma - B*(gamma*gamma);
  const x1 = 2n*C*delta*(-beta) + b0*(delta*alpha + (-beta)*(-gamma)) - 2n*B*alpha*(-gamma);
  const x0 = C*(beta*beta) + b0*(-beta)*alpha - B*(alpha*alpha);

  let g = gcdBI(gcdBI(absBI(x2), absBI(x1)), absBI(x0));
  let A2 = x2 / g, B2 = x1 / g, C2 = x0 / g;
  if (A2 < 0n) { A2 = -A2; B2 = -B2; C2 = -C2; }
  return { A: A2, B: B2, C: C2 };
};

// ---------- Formatting (including LaTeX helpers) ----------
const fmtBI = (x: bigint) => x.toString();
const fmtRational = (p: bigint, q: bigint) => `${p.toString()} / ${q.toString()}`;

// backslash builder to avoid escaping issues in source
const bs = String.fromCharCode(92);
const latexRational = (p: bigint, q: bigint) => `${bs}dfrac{${p.toString()}}{${q.toString()}}`;
const latexSurd = (P: bigint, Q: bigint, R: bigint, Delta: bigint) => {
  // Simplify display: extract square factors from Δ, reduce gcd, drop coefficient 1
  const abs = (x: bigint) => (x < 0n ? -x : x);

  // If surd part vanishes, show a reduced rational
  if (Q === 0n || Delta === 0n) {
    let g = gcdBI(abs(P), abs(R));
    let p = P / g, r = R / g;
    if (r < 0n) { p = -p; r = -r; }
    return `${bs}dfrac{${p.toString()}}{${r.toString()}}`;
  }

  // Extract the largest square factor from Delta: Delta = (s^2) * D0 with D0 squarefree
  let D = Delta;
  let s = 1n;
  for (let d = 2n; d*d <= D; d++) {
    const d2 = d*d;
    while (D % d2 === 0n) { D /= d2; s *= d; }
  }

  let P1 = P;
  let Q1 = Q * s; // √Delta = s√D
  let R1 = R;

  // Reduce common gcd among P1, Q1, R1; normalize sign so R1 > 0
  let g = gcdBI(gcdBI(abs(P1), abs(Q1)), abs(R1));
  if (g !== 0n) { P1 /= g; Q1 /= g; R1 /= g; }
  if (R1 < 0n) { P1 = -P1; Q1 = -Q1; R1 = -R1; }

  // If D==1, the surd becomes rational
  if (D === 1n) {
    let num = P1 + Q1; // √1 = 1
    let g2 = gcdBI(abs(num), abs(R1));
    if (g2 !== 0n) { num /= g2; R1 /= g2; }
    if (R1 < 0n) { num = -num; R1 = -R1; }
    return R1 === 1n ? `${num.toString()}` : `${bs}dfrac{${num.toString()}}{${R1.toString()}}`;
  }

  const absQ = Q1 < 0n ? -Q1 : Q1;
  const coeff = absQ === 1n ? "" : absQ.toString();
  let numerator: string;
  if (P1 === 0n) {
    numerator = `${Q1 < 0n ? "-" : ""}${coeff}${bs}sqrt{${D.toString()}}`;
  } else {
    const sign = Q1 < 0n ? " - " : " + ";
    numerator = `${P1.toString()}${sign}${coeff}${bs}sqrt{${D.toString()}}`;
  }
  return R1 === 1n ? numerator : `${bs}dfrac{${numerator}}{${R1.toString()}}`;
};
const latexPoly = (A: bigint, B: bigint, C: bigint) => {
  // Pretty-print: omit 1 coefficients, show '-' for -1, drop zero terms
  const s = (coef: bigint, sym: string) => {
    if (coef === 0n) return null;
    const neg = coef < 0n;
    const mag = (neg ? -coef : coef).toString();
    const body = sym ? (mag === '1' ? sym : `${mag}${sym}`) : mag; // no '1x', no '1x^{2}'
    return { neg, body };
  };
  const t2 = s(A, 'x^{2}');
  const t1 = s(B, 'x');
  const t0 = s(C, '');
  const terms = [t2, t1, t0].filter(Boolean) as {neg:boolean,body:string}[];
  if (terms.length === 0) return '0=0';
  let out = '';
  terms.forEach((t, i) => {
    if (i === 0) {
      out += (t.neg ? '-' : '') + t.body;
    } else {
      out += t.neg ? ' - ' + t.body : ' + ' + t.body;
    }
  });
  return out + ' = 0';
};

const Latex: React.FC<{ expr: string; block?: boolean }> = ({ expr, block }) => {
  const html = katex.renderToString(expr, { throwOnError: false, displayMode: !!block });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
};

// rounding
const roundTo = (x: number, d: number) => {
  const k = Math.pow(10, d);
  return Math.round((x + Number.EPSILON) * k) / k;
};

const joinSeq = (arr: bigint[]) => arr.map(v => v.toString()).join(", ");

// Compact number field with +/- buttons (no native spinners)
const NumberField: React.FC<{ label: string; value: number; setValue: (n: number) => void; min?: number; max?: number; step?: number }> = ({ label, value, setValue, min = 1, max = 50, step = 1 }) => {
  const clamp = (n: number) => Math.max(min, Math.min(max, n));
  const parseVal = (s: string) => {
    const m = s.match(/^-?[0-9]+$/);
    return m ? clamp(parseInt(s, 10)) : value;
  };
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-slate-300 w-28">{label}</label>
      <div className="flex items-stretch rounded-lg border border-slate-700 bg-slate-900/60 overflow-hidden">
        <button type="button" onClick={()=>setValue(clamp(value - step))} className="px-3 text-sm hover:bg-slate-800">−</button>
        <input type="text" inputMode="numeric" value={value}
               onChange={e=>setValue(parseVal(e.target.value))}
               onBlur={e=>setValue(parseVal(e.target.value))}
               className="w-20 appearance-none bg-transparent px-2 py-2 font-mono text-slate-100 text-sm focus:outline-none text-center" />
        <button type="button" onClick={()=>setValue(clamp(value + step))} className="px-3 text-sm hover:bg-slate-800">＋</button>
      </div>
    </div>
  );
};

// ---------- Expression parsing & arithmetic (rational / quadratic surd) ----------

// Value types
 type Rat = { kind: 'rat'; p: bigint; q: bigint };
 type ValSurd = { kind: 'surd'; P: bigint; Q: bigint; R: bigint; Delta: bigint };
 type Approx = { kind: 'approx'; x: number };
 type Val = Rat | ValSurd | Approx;

 const rat = (p: bigint, q: bigint = 1n): Rat => normRat({ kind: 'rat', p, q });
 const normRat = (r: Rat): Rat => {
   let { p, q } = r;
   if (q === 0n) throw new Error('division by zero');
   const g = gcdBI(absBI(p), absBI(q));
   p /= g; q /= g; if (q < 0n) { p = -p; q = -q; }
   return { kind: 'rat', p, q };
 };
 const toApprox = (v: Val): number => v.kind === 'rat' ? Number(v.p) / Number(v.q)
   : v.kind === 'surd' ? (Number(v.P) + Number(v.Q) * Math.sqrt(Number(v.Delta))) / Number(v.R)
   : v.x;

 const normSurd = (s: ValSurd): ValSurd => {
   // reduce gcd and normalize sign (do not factor Delta here; display will do)
   let { P, Q, R, Delta } = s;
   let g = gcdBI(gcdBI(absBI(P), absBI(Q)), absBI(R));
   if (g !== 0n) { P /= g; Q /= g; R /= g; }
   if (R < 0n) { P = -P; Q = -Q; R = -R; }
   if (Q === 0n) { // becomes rational
     return { kind: 'surd', P, Q, R, Delta };
   }
   return { kind: 'surd', P, Q, R, Delta };
 };

 // Convert CF (finite/periodic) to Val
 const cfToVal = (cf: ParsedCF): Val => {
   if (!cf.isPeriodic) {
     const { p, q } = evalFiniteExact(cf.a0, cf.prefix);
     return rat(p, q);
   }
   const s = surdFromPeriodic([cf.a0, ...cf.prefix], cf.period);
   return { kind: 'surd', P: s.P, Q: s.Q, R: s.R, Delta: s.Delta };
 };

 // Arithmetic in Q(√Δ)
 const sameDelta = (a: ValSurd, b: ValSurd) => a.Delta === b.Delta;
 const addVal = (a: Val, b: Val): Val => {
   if (a.kind === 'approx' || b.kind === 'approx') return { kind: 'approx', x: toApprox(a) + toApprox(b) };
   if (a.kind === 'rat' && b.kind === 'rat') return rat(a.p * b.q + b.p * a.q, a.q * b.q);
   if (a.kind === 'surd' && b.kind === 'rat') {
     const P = a.P * b.q + b.p * a.R; const Q = a.Q * b.q; const R = a.R * b.q;
     return normSurd({ kind:'surd', P, Q, R, Delta: a.Delta });
   }
   if (a.kind === 'rat' && b.kind === 'surd') return addVal(b, a);
   // surd + surd
   if (a.kind === 'surd' && b.kind === 'surd') {
     if (!sameDelta(a,b)) return { kind: 'approx', x: toApprox(a) + toApprox(b) };
     const P = a.P * b.R + b.P * a.R;
     const Q = a.Q * b.R + b.Q * a.R;
     const R = a.R * b.R;
     return normSurd({ kind:'surd', P, Q, R, Delta: a.Delta });
   }
   return { kind: 'approx', x: toApprox(a) + toApprox(b) };
 };
 const subVal = (a: Val, b: Val): Val => addVal(a, mulVal(rat(-1n), b));
 const mulVal = (a: Val, b: Val): Val => {
   if (a.kind === 'approx' || b.kind === 'approx') return { kind: 'approx', x: toApprox(a) * toApprox(b) };
   if (a.kind === 'rat' && b.kind === 'rat') return rat(a.p * b.p, a.q * b.q);
   if (a.kind === 'surd' && b.kind === 'rat') {
     const P = a.P * b.p; const Q = a.Q * b.p; const R = a.R * b.q; return normSurd({ kind:'surd', P, Q, R, Delta: a.Delta });
   }
   if (a.kind === 'rat' && b.kind === 'surd') return mulVal(b, a);
   // surd * surd
   if (a.kind === 'surd' && b.kind === 'surd') {
     if (!sameDelta(a,b)) return { kind: 'approx', x: toApprox(a) * toApprox(b) };
     const P = a.P * b.P + a.Q * b.Q * a.Delta;
     const Q = a.P * b.Q + b.P * a.Q;
     const R = a.R * b.R;
     return normSurd({ kind:'surd', P, Q, R, Delta: a.Delta });
   }
   return { kind: 'approx', x: toApprox(a) * toApprox(b) };
 };
 const divVal = (a: Val, b: Val): Val => {
   if (a.kind === 'approx' || b.kind === 'approx') return { kind: 'approx', x: toApprox(a) / toApprox(b) };
   if (b.kind === 'rat' && b.p === 0n) throw new Error('division by zero');
   if (a.kind === 'rat' && b.kind === 'rat') return rat(a.p * b.q, a.q * b.p);
   if (a.kind === 'surd' && b.kind === 'rat') {
     if (b.p === 0n) throw new Error('division by zero');
     const P = a.P * b.q; const Q = a.Q * b.q; const R = a.R * b.p; return normSurd({ kind:'surd', P, Q, R, Delta: a.Delta });
   }
   if (a.kind === 'rat' && b.kind === 'surd') {
     // a / b = a * (conj(b)) / (b * conj(b))
     const conj = { P: b.P, Q: -b.Q };
     const numP = a.p * (conj.P); // multiplied later by b.R in norm
     const numQ = a.p * (conj.Q);
     const den = a.q * (b.P*b.P - b.Q*b.Q*b.Delta);
     const P = numP * b.R; const Q = numQ * b.R; const R = den; return normSurd({ kind:'surd', P, Q, R, Delta: b.Delta });
   }
   if (a.kind === 'surd' && b.kind === 'surd') {
     if (!sameDelta(a,b)) return { kind: 'approx', x: toApprox(a) / toApprox(b) };
     const denom = b.P*b.P - b.Q*b.Q*b.Delta;
     if (denom === 0n) throw new Error('division by zero');
     const P = (a.P*b.P - a.Q*b.Q*b.Delta) * b.R;
     const Q = (b.P*a.Q - a.P*b.Q) * b.R;
     const R = a.R * denom;
     return normSurd({ kind:'surd', P, Q, R, Delta: a.Delta });
   }
   return { kind: 'approx', x: toApprox(a) / toApprox(b) };
 };

 // Minimal polynomial from surd value x = (P + Q√Δ)/R ⇒ (R x - P)^2 - Q^2 Δ = 0
 const polyFromSurd = (s: ValSurd) => {
   if (s.Q === 0n) return null; // rational; show blank as per UI policy
   const A = s.R * s.R;
   const B = -2n * s.P * s.R;
   const C = s.P * s.P - s.Q * s.Q * s.Delta;
   // normalize sign
   let g = gcdBI(gcdBI(absBI(A), absBI(B)), absBI(C));
   let A2 = A / g, B2 = B / g, C2 = C / g;
   if (A2 < 0n) { A2 = -A2; B2 = -B2; C2 = -C2; }
   return { A: A2, B: B2, C: C2 };
 };

 // -------- Tokenizer & Parser for expressions --------
 type Tok = { t: 'num'; v: bigint } | { t: 'cf'; s: string } | { t: '+'|'-'|'*'|'/'|'('|')' };
 const tokenize = (src: string): Tok[] => {
   const s = src.trim();
   const toks: Tok[] = [];
   let i = 0;
   const peek = () => s[i];
   while (i < s.length) {
     const c = peek();
     if (c <= ' ') { i++; continue; }
     if (c === '[') {
       let j = i+1; let depth = 1;
       while (j < s.length && depth > 0) {
         const ch = s[j];
         if (ch === '[') depth++; else if (ch === ']') depth--;
         j++;
       }
       if (depth !== 0) throw new Error('Unclosed [ ... ]');
       const frag = s.slice(i, j); // includes brackets
       toks.push({ t: 'cf', s: frag });
       i = j; continue;
     }
     if ('+-*/()'.includes(c)) { toks.push({ t: c as any }); i++; continue; }
     if (/[0-9]/.test(c)) {
       let j = i+1; while (j < s.length && /[0-9]/.test(s[j])) j++;
       toks.push({ t:'num', v: BI(s.slice(i, j)) }); i = j; continue;
     }
     throw new Error(`Unexpected character: ${c}`);
   }
   // implied multiplication: num][ or num( or )[
   const out: Tok[] = [];
   for (let k=0; k<toks.length; k++) {
     const a = toks[k]; const b = toks[k+1];
     out.push(a);
     if (!b) break;
     const aCat = (a.t === 'num' || a.t === 'cf' || a.t === ')');
     const bCat = (b.t === 'cf' || b.t === '(');
     if (aCat && bCat) out.push({ t: '*' });
   }
   return out;
 };

 // Recursive descent parser
 const parseExpression = (src: string): Val => {
   const toks = tokenize(src);
   let k = 0;
   const at = () => toks[k];
   const eat = (t?: Tok['t']) => { const x = toks[k]; if (!x) throw new Error('Unexpected end'); if (t && x.t !== t) throw new Error(`Expected ${t}`); k++; return x; };

   const parseFactor = (): Val => {
     let sign = 1n;
     while (at() && (at()!.t === '+' || at()!.t === '-')) { sign *= (eat().t === '+' ? 1n : -1n); }
     let res: Val;
     const tok = at(); if (!tok) throw new Error('Expected factor');
     if (tok.t === 'num') { eat(); res = rat(tok.v); }
     else if (tok.t === 'cf') { eat(); res = cfToVal(parseCF(tok.s)); }
     else if (tok.t === '(') { eat('('); res = parseExpr(); eat(')'); }
     else throw new Error('Invalid factor');
     if (sign === -1n) res = mulVal(rat(-1n), res);
     return res;
   };
   const parseTerm = (): Val => {
     let res = parseFactor();
     while (at() && (at()!.t === '*' || at()!.t === '/')) {
       const op = eat().t; const rhs = parseFactor();
       res = (op === '*') ? mulVal(res, rhs) : divVal(res, rhs);
     }
     return res;
   };
   const parseExpr = (): Val => {
     let res = parseTerm();
     while (at() && (at()!.t === '+' || at()!.t === '-')) {
       const op = eat().t; const rhs = parseTerm();
       res = (op === '+') ? addVal(res, rhs) : subVal(res, rhs);
     }
     return res;
   };

   const v = parseExpr();
   if (k !== toks.length) throw new Error('Unexpected token at end');
   return v;
 };

 const looksLikeExpression = (src: string): boolean => {
   // detect operators outside [...] or implied num[...
   let depth = 0;
   for (let i=0; i<src.length; i++) {
     const c = src[i];
     if (c === '[') depth++; else if (c === ']') depth = Math.max(0, depth-1);
     if (depth === 0 && '+-*/'.includes(c)) return true;
     if (depth === 0 && /[0-9\)]/.test(c) && src.slice(i+1).trimStart().startsWith('[')) return true;
   }
   return false;
 };

// ---------- UI Component ----------
export default function ContinuedFractionCalculator() {
  const [input, setInput] = useState<string>("[2; 1, 2, 3, 1]");
  const [digits, setDigits] = useState<number>(12);
  const [convCount, setConvCount] = useState<number>(12);

  const parsed = useMemo(() => {
    // Skip CF parsing entirely when the input is an arithmetic expression.
    if (looksLikeExpression(input)) {
      return { ok: false as const, error: "" as string };
    }
    try {
      const p = parseCF(input);
      return { ok: true as const, value: p };
    } catch (e: any) {
      return { ok: false as const, error: e.message as string };
    }
  }, [input]);

  const result = useMemo(() => {
    // Expression mode
    if (looksLikeExpression(input)) {
      try {
        const v = parseExpression(input);
        if (v.kind === 'rat') {
          return { kind: 'expr', expr: { exactLatex: latexRational(v.p, v.q), approx: Number(v.p)/Number(v.q), poly: null } } as const;
        } else if (v.kind === 'surd') {
          const poly = polyFromSurd(v);
          return { kind: 'expr', expr: { exactLatex: latexSurd(v.P, v.Q, v.R, v.Delta), approx: (Number(v.P)+Number(v.Q)*Math.sqrt(Number(v.Delta)))/Number(v.R), poly } } as const;
        } else {
          return { kind: 'expr', expr: { exactLatex: `${bs}approx ${v.x}`, approx: v.x, poly: null } } as const;
        }
      } catch (e: any) {
        return { kind: 'error', error: e.message || String(e) } as const;
      }
    }

    if (!parsed.ok) return null;
    const { a0, prefix, period, isPeriodic } = parsed.value;

    if (!isPeriodic) {
      const { p, q } = evalFiniteExact(a0, prefix);
      const approx = Number(p) / Number(q);
      const conv = convergents([a0, ...prefix]);
      return {
        kind: "finite" as const,
        p, q, approx, conv
      };
    }

    // Periodic
    const Mpref = matFromSeq([a0, ...prefix]);
    const [alpha, beta] = Mpref[0];
    const [gamma, delta] = Mpref[1];
    const Mr = matFromSeq(period);
    const [A, B] = Mr[0];
    const [C, D] = Mr[1];

    const surd = surdFromPeriodic([a0, ...prefix], period);
    const poly = minimalPolynomial(alpha, beta, gamma, delta, A, B, C, D);

    const showLen = Math.max(1, convCount);
    const expanded: bigint[] = [a0, ...prefix];
    while (expanded.length < showLen) expanded.push(...period);
    const conv = convergents(expanded.slice(0, showLen));

    return {
      kind: "periodic" as const,
      surd, poly, period, prefixAll: [a0, ...prefix], conv
    };
  }, [parsed, convCount, input]);

  const onExample = (str: string) => setInput(str);

  const copy = async (text: string) => {
    try { await navigator.clipboard.writeText(text); alert("Copied!"); }
    catch { alert("Copy failed"); }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-900 via-slate-950 to-black text-slate-100 py-10 px-4">
      <style>{`
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
      `}</style>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-2">Continued Fraction Calculator</h1>
        <div className="text-slate-300 mb-6 space-y-1">
          <div>Parse continued fractions in the form <span className="font-mono">[a0; a1, a2, ...]</span>.</div>
          <div>Supports periodic parts via <span className="font-mono">(…)</span>, <span className="font-mono">|</span>, or trailing <span className="font-mono">...</span>.</div>
          <div>Shows the exact value in LaTeX (rational or quadratic surd), approximation, and convergents.</div>
        </div>

        <div className="bg-white/5 backdrop-blur rounded-2xl p-4 sm:p-6 shadow-xl ring-1 ring-white/10">
          <label className="block text-sm mb-2 text-slate-300">
            Examples: <span className="font-mono">[2; 1, 2, 3, 1]</span>, <span className="font-mono">[1; 2, 3, (1, 2)]</span>, <span className="font-mono">[0; 1,2,1,2,...]</span>
          </label>
          <div className="flex flex-col sm:flex-row gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-1 rounded-xl bg-slate-900/60 border border-slate-700 px-4 py-3 font-mono text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              placeholder="[a0; a1, a2, ...]"
            />
            <div className="flex items-center gap-2">
              <button onClick={() => onExample("[2; 1, 2, 3, 1]")} className="px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 font-mono text-sm">Ex1</button>
              <button onClick={() => onExample("[1; 1, 1, ...]")} className="px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 font-mono text-sm">φ</button>
              <button onClick={() => onExample("[1; (2)]")} className="px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 font-mono text-sm">√2</button>
              <button onClick={() => onExample("[0; 1, 2, 1, 2, ...]")} className="px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 font-mono text-sm">period 1,2</button>
            </div>
          </div>

          {/* Digits: compact control */}
          <div className="mt-4 grid sm:grid-cols-2 gap-3">
            <NumberField label="Digits" value={digits} setValue={setDigits} min={4} max={20} />
            <NumberField label="Convergents" value={convCount} setValue={setConvCount} min={1} max={50} />
          </div>
        </div>

        {/* Results */}
        <div className="mt-6 grid gap-6">
          {!looksLikeExpression(input) && !parsed.ok && (
            <div className="bg-rose-900/40 border border-rose-700 text-rose-100 rounded-xl p-4">
              <div className="font-bold">Parsing Error</div>
              <div className="opacity-90 mt-1">{parsed.error}</div>
            </div>
          )}

          {result && result.kind === "error" && (
            <div className="bg-rose-900/40 border border-rose-700 text-rose-100 rounded-xl p-4">
              <div className="font-bold">Parsing Error</div>
              <div className="opacity-90 mt-1">{(result as any).error}</div>
            </div>
          )}

          {result && result.kind === "expr" && (
            <div className="bg-white/5 backdrop-blur rounded-2xl p-5 shadow-xl ring-1 ring-white/10">
              <h2 className="text-xl font-semibold mb-3">Expression result</h2>
              <div className="grid lg:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-slate-300">Exact value (LaTeX)</div>
                    <div className="mt-1 text-lg break-all"><Latex expr={(result as any).expr.exactLatex} block /></div>
                    <div className="mt-2 flex gap-2">
                      <button onClick={()=>copy((result as any).expr.exactLatex)} className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-sm">Copy LaTeX</button>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-300">Approximation</div>
                    <div className="font-mono text-lg mt-1">{roundTo((result as any).expr.approx, digits)}</div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-slate-300">Minimal polynomial (in x)</div>
                    <div className="mt-1 text-lg">{(result as any).expr.poly ? <Latex expr={latexPoly((result as any).expr.poly.A, (result as any).expr.poly.B, (result as any).expr.poly.C)} block /> : <span className="text-slate-400">—</span>}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-300 mb-2">Convergents</div>
                    <div className="text-slate-400">—</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {parsed.ok && result && result.kind === "finite" && (
            <div className="bg-white/5 backdrop-blur rounded-2xl p-5 shadow-xl ring-1 ring-white/10">
              <h2 className="text-xl font-semibold mb-3">Finite continued fraction</h2>
              <div className="grid lg:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-slate-300">Exact value (LaTeX)</div>
                    <div className="mt-1 text-lg"><Latex expr={latexRational(result.p, result.q)} block /></div>
                    <div className="mt-2 flex gap-2">
                      <button
                        onClick={()=>copy(latexRational(result.p, result.q))}
                        className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-sm"
                      >
                        Copy LaTeX
                      </button>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-300">Approximation</div>
                    <div className="font-mono text-lg mt-1">{roundTo(result.approx, digits)}</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-slate-300">Minimal polynomial (in x)</div>
                    <div className="mt-1 text-lg text-slate-400">—</div>
                  </div>

                  <div>
                    <div className="text-sm text-slate-300 mb-2">Convergents (first {Math.min(result.conv.length, convCount)})</div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="text-slate-300">
                            <th className="py-2 pr-4">k</th>
                            <th className="py-2 pr-4">p_k / q_k</th>
                            <th className="py-2">value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.conv.slice(0, convCount).map((c, i) => (
                            <tr key={i} className="border-t border-white/10">
                              <td className="py-2 pr-4 tabular-nums">{i}</td>
                              <td className="py-2 pr-4 font-mono">{fmtRational(c.p, c.q)}</td>
                              <td className="py-2 font-mono">{roundTo(Number(c.p)/Number(c.q), digits)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {parsed.ok && result && result.kind === "periodic" && (
            <div className="bg-white/5 backdrop-blur rounded-2xl p-5 shadow-xl ring-1 ring-white/10">
              <h2 className="text-xl font-semibold mb-3">Periodic continued fraction (infinite)</h2>

              <div className="grid lg:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-slate-300">Detected structure</div>
                    <div className="font-mono mt-1 text-slate-200">
                      [ {fmtBI(result.prefixAll[0])}; {joinSeq(result.prefixAll.slice(1))} | {joinSeq(result.period)} ]
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-300">Exact value (LaTeX)</div>
                    <div className="mt-1 text-lg break-all">
                      <Latex expr={latexSurd(result.surd.P, result.surd.Q, result.surd.R, result.surd.Delta)} block />
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      Δ = {result.surd.Delta.toString()}
                      {/* or: <Latex expr={`${bs}Delta = ${result.surd.Delta.toString()}`} /> */}
                    </div>
                    <div className="mt-2 flex gap-2">
                      <button
                        onClick={()=>copy(latexSurd(result.surd.P, result.surd.Q, result.surd.R, result.surd.Delta))}
                        className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-sm"
                      >
                        Copy LaTeX
                      </button>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-300">Approximation</div>
                    <div className="font-mono text-lg mt-1">{roundTo(result.surd.approx, digits)}</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-slate-300">Minimal polynomial (in x)</div>
                    <div className="mt-1 text-lg">
                      <Latex expr={latexPoly(result.poly.A, result.poly.B, result.poly.C)} block />
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-slate-300">Convergents (first {Math.min(result.conv.length, convCount)})</div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="text-slate-300">
                            <th className="py-2 pr-4">k</th>
                            <th className="py-2 pr-4">p_k / q_k</th>
                            <th className="py-2">value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.conv.slice(0, convCount).map((c, i) => (
                            <tr key={i} className="border-t border-white/10">
                              <td className="py-2 pr-4 tabular-nums">{i}</td>
                              <td className="py-2 pr-4 font-mono">{fmtRational(c.p, c.q)}</td>
                              <td className="py-2 font-mono">{roundTo(Number(c.p)/Number(c.q), digits)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Help */}
          <div className="bg-white/5 backdrop-blur rounded-2xl p-5 shadow-xl ring-1 ring-white/10">
            <h3 className="text-lg font-semibold mb-2">How to write input</h3>
            <ul className="list-disc pl-6 space-y-1 text-slate-300">
              <li><span className="font-mono">[a0; a1, a2, ..., an]</span> — finite continued fraction</li>
              <li><span className="font-mono">[a0; prefix, (b1, b2, ...)]</span> — content in <span className="font-semibold">()</span> repeats</li>
              <li><span className="font-mono">[a0; prefix | b1, b2, ...]</span> — right side of <span className="font-semibold">|</span> repeats</li>
              <li><span className="font-mono">[a0; 1, 2, 1, 2, ...]</span> — trailing <span className="font-semibold">...</span> to auto-detect the last repeating block</li>
            </ul>
          </div>
        </div>

        <footer className="mt-10 text-center text-slate-500 text-xs">
          Made by ChatGPT with Koki Yamada
        </footer>
      </div>
    </div>
  );
}
