# QGEMM Canonical Form

Purpose: define the tensor-level writing convention for expressing QGEMM in Triton so that later compiler work can recognize its hierarchical scaling structure and recover each operand's real memory layout.

Audience: users writing QGEMM kernels and compiler developers implementing recognition, pass organization, and loop scheduling and data layout work.

Authoritative source: `reports/开题报告.md`, sections 4.1.1, 4.1.2, and 4.1.3, in research content 1, 「CPU 上跨平台性能可移植的 QGEMM 编程系统设计」. The proposal takes precedence over steering documents when they disagree.

Reading conventions:

- Dimensions are compile-time constants unless stated otherwise.
- Numbered steps refer to the sequence in section 2.
- Future compiler obligations do not describe an implemented pass.

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| `M`, `N`, `K` | Logical operator size: activation `(M, K)` and weight `(N, K)` produce `(M, N)`; `K` is the reduction dimension |
| `T` | Number of programs, with `N % T == 0` |
| `m`, `n`, `k` | Per-program logical extents: `m = M`, `n = N // T`, `k = K` |
| `p` | A path of the reduction DAG, running from its leaves to a merge point or to the final reduction |
| `L^p` | Number of levels on the divisibility chain of path `p`, `L^p >= 1`; the superscript is dropped, leaving `L`, `QK_j`, `r_j`, `G_j`, wherever only one path is under discussion |
| `QK_j^p` | Quantization block size of level `j` on path `p`, with `QK_0^p = 1` |
| `r_j^p` | `QK_j^p / QK_{j-1}^p`; therefore `r_1^p = QK_1^p` |
| `G_j^p` | `K / QK_j^p`, the reduction length left after level `j` of path `p`; `G_0^p = K` |
| `C(v)` | Cumulative quantization block size at position `v`: the element count already reduced there, equal to the product of the split ratios of the levels of that path before `v` |
| `j_merge`, `sign` | The level at which two paths merge, and the sign, `+` or `-`, written there |

## 2. Compute Sequence

The canonical form is a property of the whole reduction DAG leading from the input components to the output `(m, n)`. Every path of that DAG has the same shape:

1. Every leaf operand is broadcast to `(m, n, k)`.
2. Two broadcast leaves are multiplied element-wise, producing `(m, n, k)`. A degenerate path has a single leaf and no element-wise multiply, so this step is absent there.
3. For each level `j = 1, ..., L^p` of the path's divisibility chain, in ascending block-size order: (a) a split-direction `tl.reshape` splits the current reduction dimension as `(G_j^p, r_j^p)`, with `r_j^p` innermost; (b) that innermost dimension is reduced; (c) the level's factors are applied.
4. Paths may merge, each merge joining two operands at any level through one addition or subtraction.
5. The whole DAG ends in one reduction that carries no factor and produces `(m, n)`.

```mermaid
flowchart TD
    L1["P1 leaves: broadcast a, broadcast w"] --> M1["element-wise multiply -> (m,n,k)"]
    M1 --> R1["P1 level 1: split reshape, reduce, factors -> (m,n,G_1)"]
    L2["P2 leaf: broadcast a"] --> R2["P2 level 1: split reshape, reduce, factors -> (m,n,G_1)"]
    R1 --> R1B["P1 level 2: split reshape, reduce, factors -> (m,n,G_2)"]
    R2 --> R2B["P2 level 2: split reshape, reduce, factors -> (m,n,G_2)"]
    R1B --> MG["merge at level 2, sign + or -"]
    R2B --> MG
    MG --> OUT["final reduction, no factor -> (m,n)"]
```

The one legality condition on a merge is that both operands carry the same cumulative quantization block size `C` there, equivalently the same current shape. Neither the level index nor the inner split structure of the two paths has to agree. A path whose factors exist only at the outer level may skip the inner split and reduce 256 elements in one step: its level index is 1 where the main path's is 2, and the merge stays legal, because both operands have reduced 256 elements. Deciding by equal level indices, or by walking the two chains and demanding the two paths agree on every level's block size, rejects that writing wrongly.

| Path-internal step | Operation | Shapes | Stage |
|--------------------|-----------|--------|-------|
| leaf | weight decompression, or a plain load on a side already stored as the integer value | stored components -> `a: (m, k)` or `w: (n, k)` | data preparation |
| broadcast | `tl.expand_dims`, `tl.broadcast_to` | `(m, k)` or `(n, k)` -> `(m, n, k)` | matrix multiplication |
| element-wise multiply | product casts and multiply of the two broadcast leaves | two `(m, n, k)` -> `(m, n, k)` | matrix multiplication |
| level `j` split and reduce | split-direction `tl.reshape`, then reduce the innermost dimension | `(m, n, G_{j-1}) -> (m, n, G_j, r_j) -> (m, n, G_j)` | matrix multiplication |
| level `j` split and reduce, degenerate path | the block sum of a zero-point branch, which is level 1 of that path | `(m, n, k) -> (m, n, G_j, r_j) -> (m, n, G_j)` | matrix multiplication |
| level `j` factors | apply this level's factors to this level's reduction result | `(m, n, G_j) -> (m, n, G_j)` | scaling |
| merge | one addition or subtraction of two paths with equal `C` | two `(m, n, G_j) -> (m, n, G_j)` | scaling |
| final reduction | outermost plain reduction, carrying no factor | `(m, n, G_L) -> (m, n)` | scaling |

### Single-path instance

A format whose DAG is a single path with no merge point instantiates the above as the following numbered sequence. The `# step` comments of the verification kernel in section 10 and the `// step` annotations of the measured IR in section 12 are the step numbers of this single-path instance, corresponding one-to-one to the path-internal steps above.

1. The data preparation stage decompresses the stored weight into the integer operand `w: (n, k)`; the activation enters as the integer operand `a: (m, k)`.
2. The matrix multiplication stage inserts the missing parallel dimension with `tl.expand_dims` and broadcasts both operands to `(m, n, k)` with `tl.broadcast_to`, as written out in section 10.
3. After conversion to the product type, the two three-dimensional operands are multiplied element-wise, producing `(m, n, k)`.
4. For each level `j = 1, ..., L` in ascending block-size order: (a) split the current reduction dimension as `(G_j, r_j)` with `r_j` innermost; (b) reduce that innermost dimension; (c) apply the level's factors to the reduction result.
5. Reduce the remaining dimension `G_L` without a factor, producing `(m, n)`.

For `L = 1` the levels expand as `(m,n,K) -> (m,n,K/QK_1,QK_1) -> (m,n,K/QK_1) -> (m,n,K/QK_1) -> (m,n)`.

For `L = 2` with `K = 512`, `QK_1 = 32`, `QK_2 = 256`, hence `G_1 = 16`, `G_2 = 2`, `r_2 = 8`: `(m,n,512) -> (m,n,16,32) -> (m,n,16) -> (m,n,16) -> (m,n,2,8) -> (m,n,2) -> (m,n,2) -> (m,n)`.

No path contains `tl.dot`. A writing that splits the quantization-block dimensions before broadcasting and directly broadcasts four- or five-dimensional tensors, as existing kernels under `benchmark/kernels/` do, is not the canonical form.

## 3. Divisibility Chain

The chain is per path. Path `p` reads its own chain from its own split reshapes: if their innermost extents, written from inner to outer, are `r_1^p, ..., r_{L^p}^p`, then

```
QK_j^p = r_1^p * ... * r_j^p
```

Chain order is therefore the order in which that path's split reshapes are written from the innermost quantization level outward, block sizes strictly increase along the chain, and each block size is an integer multiple of the previous one. `K` is an integer multiple of `QK_{L^p}^p`, and consequently of every level of that path.

Within one path the levels collect the quantization levels of both its leaves: equal block sizes on the two sides become one level, unequal sizes stay separate levels. A factor may exist on only one side at a level; the other side then contributes the constant factor 1, leaving that level's shape unchanged.

Two paths of the same DAG need not agree. Their numbers of levels may differ, and their level `j` may carry different block sizes. Nothing ties two chains together except a merge point, where the two operands must have the same cumulative quantization block size `C`, equivalently the same current shape; merging two intermediate results with unequal `C` is not the canonical form. That single condition is the reduction-DAG form of forbidding a factor to move across levels: a factor belongs to the level whose reduction produced the value it multiplies, and joining values that have reduced different element counts would be the only way to slide it elsewhere.

For Q4_K x Q8_K the DAG has two paths, each with the chain `{32, 256}`:

| Level | Path | Block size and ratio | Factor sources | Reduction dimension |
|-------|------|----------------------|----------------|---------------------|
| 1 | main | `QK_1 = r_1 = 32` | weight `sc`; activation factor is 1 | `K -> K/32` |
| 1 | zero point | `QK_1 = r_1 = 32` | weight `m`; activation factor is 1 | `K -> K/32` |
| 2 | main | `QK_2 = 256`, `r_2 = 8` | weight `d`; activation `a_d` | `K/32 -> K/256` |
| 2 | zero point | `QK_2 = 256`, `r_2 = 8` | weight `dmin`; activation `a_d` | `K/32 -> K/256` |

The two paths merge after level 2, where both carry `C = 256`, with the sign the user wrote there, a minus for Q4_K.

## 4. Factor Invariance and Branch Linearity

Factor invariance is the one load-bearing condition of the canonical form: a factor does not vary along the dimension reduced at its level. Its algebraic basis is `sum_r (s * x_r) = s * sum_r x_r`, which holds exactly when `s` does not vary with `r`, and it is readable straight off Linalg: the factor operand's indexing map over that level's iteration domain does not contain the reduced dimension.

Applying a factor before the reduction and applying it afterwards are therefore two writings of one thing, and recognition accepts both. A factor tensor whose rank equals that of the level's reduction result `(m, n, G_j)` is the post-reduction special case: the reduced dimension is not in the iteration domain at all, so invariance holds automatically. A factor applied before the reduction is equally invariant as long as its map omits that dimension. A factor absent from one side is the constant 1.

Two factors at one level, one per side, may be multiplied together and applied once, or applied in sequence; invariance plus commutativity of multiplication makes the two forms equivalent. The choice is the user's because multiplication order affects floating-point precision, and compiler transformations preserve the order written:

```python
level = red.to(tl.float32) * (a_d.to(tl.float32) * w_d.to(tl.float32))   # factors first
level = red.to(tl.float32) * a_d.to(tl.float32) * w_d.to(tl.float32)     # or in sequence
```

Branch linearity makes a zero point an ordinary path rather than a special case. Its leaf is a single broadcast operand, readable as a degenerate element-wise multiply against the constant 1; its per-level factors are that format's zero-point factors, `m` at level 1 and `dmin` at level 2 for Q4_K; and it joins the main path at one level through a subtraction. The sign is the one the user wrote, and recognition records only whether a merge point adds or subtracts. A zero point on each of the two operands is then expressible directly: `(a_q - a_z)(w_q - w_z)` expands into four paths merged with `+` and `-`, each path satisfying the invariant of section 2 on its own.

## 5. Cast Positions

The following positions are writing-style recommendations that make type boundaries visible:

| Type role | Suggested position |
|-----------|--------------------|
| product type | convert both three-dimensional operands immediately before the element-wise multiply |
| accumulation type | convert the summand consumed by each level's block reduction |
| scaled type | convert each reduction result immediately before that level's factors are applied |

The user chooses each concrete type freely from Triton's scalar types. Each level of each path may choose its accumulation and scaled types independently.

An identity conversion may be omitted. The outermost reduction uses the element type produced by the last scaling level unless the user explicitly converts for output storage.

These positions do not impose bit-width relationships or require a floating-point scaled type. Avoiding overflow and choosing adequate precision remain the user's responsibility.

## 6. Decompression Region

The decompression region is the computation that turns stored codes into the integer values entering the matrix multiplication. It may contain element-wise arithmetic, bit manipulation, table lookup, and dimension transformations, but it contains no reduction. Its boundary is found by walking upstream from the integer operand entering step 2 to the loads of kernel pointer arguments.

A quantized tensor enters a Triton kernel as a set of planar component tensors because Triton has no structure type. For example, Q4_K weight components include `q`, `d`, `dmin`, and `sc`, each supplied by a separate pointer, so a decompression region may have several sources. A codebook lookup is expressed as an indirect `tl.load` with the codebook table as an additional input to the region, and the index computation remains part of the path from the decoded value to its source components.

Dimension transformations may express the format's stored element order. `tl.cat` permits the joined elements to be reordered, whereas `tl.join` preserves their order; the kernel author chooses between them according to the format's required semantics. A side that is already stored as the integer value used by step 2 has a degenerate region consisting only of its load and any desired conversion. Decompression comes first, and the layout convention of section 7 governs what it produces rather than what it reads: a component that has to be decoded is loaded and reshaped in whatever way its stored form makes convenient, and neither that load nor the shape transformations on the way to the decoded value have to declare a recoverable layout, because the stored form is fixed by the quantization format rather than chosen by the user, and the weight-side layout the compiler needs is the one dynamic pre-decompression writes. What the region owes is the shape of its output: `w: (n, k)` on the weight side, and `(m, G_j)` or `(n, G_j)` for a level `j` factor that factor-side decoding derives.

Factor-side decoding is the same kind of computation on the factor side: it turns stored factor components into a level's factor tensor. Q4_K packs its 6-bit `sc` and `m` into bytes that have to be unpacked, and iq2_xxs derives its sub-block factor through a computation of the `(0.5 + s) * 0.25` kind. Both belong to the data preparation stage, sit upstream of the element-wise multiply alongside the weight decompression region, and their boundary is found the same way: walk upstream from the factor operand along the data flow to the loads of kernel pointer arguments.

That attribution does not disturb factor invariance. Level `j` factor-side decoding produces the level's factor shape, `(m, G_j)` on the activation side or `(n, G_j)` on the weight side, and that shape does not contain `r_j`, the dimension reduced at level `j`; the decoding itself contains no reduction. Invariance therefore depends only on the factor's shape, never on the computation that produced it, however long the decoding chain runs.

## 7. Layout Expression Chains

Each input component that reaches the computation without decoding — the activation quantized values, and a factor that is loaded rather than derived — expresses its real memory layout with the chain `multidimensional tl.make_block_ptr + tl.load -> a sequence of tl.trans and tl.reshape steps -> the logical shape`. The sequence may have any finite length, and there are only two conditions on it: every step's arguments are compile-time constants, and the whole sequence composes into one static permutation followed by one fold, from which the physical dimension order, each dimension's length, and each dimension's stride are still recoverable. Neither transformation is restricted to a single occurrence. A physical layout whose logical dimension order appears only once two of the loaded dimensions have been split apart is written `reshape -> trans -> reshape`: the first reshape exposes the split, the permutation puts the pieces in logical order, and the second reshape folds them back. Limiting each transformation to one occurrence would reject that writing for no reason. A component that has to be decoded is governed by section 6 instead, which constrains the shape its decoding produces rather than the way its stored form is loaded.

`tl.make_block_ptr` declares the real physical dimension order through `shape`, the element stride of each dimension through `strides`, and row-major traversal through descending `order = (rank-1, ..., 0)`. These values, the permutation, and the reshape targets are statically evaluable from literals and `tl.constexpr` values. `tl.trans` maps physical dimension order to logical dimension order, and `tl.reshape` then folds adjacent dimensions to that component's own logical shape: `(m, k)` or `(n, k)` for quantized values that need no decoding, and `(m, G_j)` on the activation side or `(n, G_j)` on the weight side, with `G_j = K / QK_j`, for a level `j` factor that is loaded rather than derived. A zero-point component has the same shape as its level's scale factor component and runs the same chain. Identity permutation or fold steps may be omitted.

For a weight stored physically as `(k/QK, QK, n)` but used logically as `(n, k)`:

```python
w_block = tl.make_block_ptr(base=w_ptr, shape=(K // QK, QK, N), strides=(QK * N, N, 1),
                            offsets=(0, 0, n_start), block_shape=(K // QK, QK, n_per_pid),
                            order=(2, 1, 0))
w = tl.load(w_block)                 # (K/QK, QK, n)
w = tl.trans(w, 2, 0, 1)             # (n, K/QK, QK)
w = tl.reshape(w, (n_per_pid, K))     # (n, k)
```

| Recovered fact | Static source | Value |
|----------------|---------------|-------|
| physical order and lengths | `shape` | `(kb, qk, n)`, `(K/QK, QK, N)` |
| physical strides | `strides` | `(QK*N, N, 1)` |
| row-major traversal | `order` | physical dimensions `2, 1, 0` from fastest to slowest |
| logical `n` | permutation entry 0 | physical dimension 2, stride 1 |
| logical `k` | permutation entries 1 and 2 plus reshape | physical dimensions 0 and 1 folded to `K` |

The output uses the mirror chain `tl.reshape(physical shape) -> tl.trans(logical-to-physical permutation) -> multidimensional tl.make_block_ptr + tl.store`: the reshape splits logical dimensions into physical dimensions, and the permutation puts them in physical order before one store. When the real layout already matches logical `(m, n)`, both transformations are identities and may be omitted:

```python
c_block = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1),
                            offsets=(0, n_start), block_shape=(M, n_per_pid), order=(1, 0))
tl.store(c_block, c)
```

This output convention gives WP B5 the information needed to combine result write-back with the user's output layout adaptation.

## 8. Thread-Level Partition

The partition described here is the reference writing used by the verified example of section 10, not a condition of the canonical form: a one-dimensional grid of `T` programs in which program `pid = tl.program_id(0)` computes the non-overlapping N-direction slice `[n_start, n_start + n_per_pid)` of output `(M, N)`, takes M and K at full extent, and contains no user loop and no user-written register-level tiling.

```python
pid = tl.program_id(0)
n_per_pid: tl.constexpr = N // T
n_start = pid * n_per_pid

tl.device_assert(tl.num_programs(0) == T, "grid.x must equal T")
tl.device_assert(tl.num_programs(1) == 1, "grid.y must be 1")
tl.device_assert(tl.num_programs(2) == 1, "grid.z must be 1")
tl.device_assert(tl.num_programs(0) * n_per_pid == N, "N must be divisible by T")
```

Weight and output block pointers use `n_start` and `n_per_pid` along N. Activation, weight, and component accesses express M and K at full extent, with no M- or K-direction program slice.

Recognition is not conditioned on this partition. The invariant of section 2 holds of the reduction DAG inside a single program, and the partition only decides which `(m, n, k)` slice a program receives; enumerating the paths upstream from the final reduction reads the already-sliced shapes. No legality condition of sections 2, 3 and 4 mentions the grid, the sliced dimension, or the number of programs. A two-dimensional grid, a partition along M, and other partitions are therefore accepted writings, untested rather than non-conforming; section 13 records them as such.

Absence of a user loop and of user-written register-level tiling is likewise part of the reference writing. One program's N slice is the outermost unit for further compiler tiling; its width does not constrain the innermost tile size. Those decisions belong to Phase B and Phase C. The three-dimensional product of step 3 is a logical tensor, whose physical working-set size Phase B controls through tiling.

## 9. Numeric Contract

| Transformation | Verdict | Boundary |
|----------------|---------|----------|
| reassociate additions within one reduction dimension | allowed | reassociation does not cross a level's factor application |
| tile parallel or reduction dimensions | allowed | each K tile boundary is an integer multiple of the quantization block size of every level of every path |
| join two intermediate results | allowed at a merge point | both operands carry the same cumulative quantization block size `C` there, equivalently the same current shape; equal level indices and equal inner split structure are not required |
| change a product, accumulation, or scaled bit width | forbidden | compiler transformations preserve the user-selected types |

Equal `C` at every merge point is what holds each factor at its own level: a factor stays with the reduction result it multiplies, and joining values that have reduced different element counts would be the only way to slide it elsewhere. The sign of each merge point and the level it sits at are the user's decisions, and compiler transformations preserve both, as they preserve the multiplication order of the factors at one level.

Allowed transformations may reorder floating-point additions, so correctness tests compare floating-point results with `atol` and `rtol` rather than requiring bit equality. When both product and accumulation types are integer types, the corresponding integer reduction results are compared bit-exactly.

## 10. Verified Example: Q8_0 × Q8_0

Q8_0 stores 32 signed 8-bit values and one `float16` scale per block. Both operands use Q8_0, so the merged divisibility chain has one level with factors on both sides. The module-level constants used below are `QK8_0 = 32` and `TL_QK8_0 = tl.constexpr(QK8_0)`.

```text
components: `a_q: int8 (m,k)`, `a_d: float16 (m,G_1)`, `w_q: int8 (n,k)`, `w_d: float16 (n,G_1)`
chain: `{32: both}`, with `L = 1`, `QK_1 = r_1 = 32`, and `G_1 = K / 32`
factors: level 1 uses activation `a_d` and weight `w_d`
region: each side is a degenerate decompression region from its quantized-value load to the integer operand entering step 2
```

The `# step` comments and adjacent shape comments are the step-by-step shape ledger; their numbers are those of the single-path instance of section 2, one per path-internal step there, because this format's DAG is a single path with no merge point. The grid is the reference partition of section 8, and the layout expression chains of section 7 are identities here because physical order equals logical order, so `tl.trans` and the layout `tl.reshape` are omitted. The body below is reproduced from `benchmark/kernels/q80_q80_strict.py`, with the block-pointer argument lists re-wrapped onto fewer lines and the step comments condensed, and nothing else changed.

```python
@triton.jit
def q80_q80_strict(
    a_q_ptr,    # int8    (M, K)
    a_d_ptr,    # float16 (M, K // QK)
    w_q_ptr,    # int8    (N, K)
    w_d_ptr,    # float16 (N, K // QK)
    c_ptr,      # float32 (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
):
    QK: tl.constexpr = TL_QK8_0
    KB: tl.constexpr = K // QK  # G_1, the reduction length left after level 1

    pid = tl.program_id(0)
    n_per_pid: tl.constexpr = N // T
    n_start = pid * n_per_pid

    tl.device_assert(tl.num_programs(0) == T, "grid.x must equal T")
    tl.device_assert(tl.num_programs(1) == 1, "grid.y must be 1")
    tl.device_assert(tl.num_programs(2) == 1, "grid.z must be 1")
    tl.device_assert(tl.num_programs(0) * n_per_pid == N, "N must be divisible by T")

    a_q_block_ptr = tl.make_block_ptr(base=a_q_ptr, shape=(M, K), strides=(K, 1),
                                      offsets=(0, 0), block_shape=(M, K), order=(1, 0))
    a_d_block_ptr = tl.make_block_ptr(base=a_d_ptr, shape=(M, KB), strides=(KB, 1),
                                      offsets=(0, 0), block_shape=(M, KB), order=(1, 0))
    w_q_block_ptr = tl.make_block_ptr(base=w_q_ptr, shape=(N, K), strides=(K, 1),
                                      offsets=(n_start, 0), block_shape=(n_per_pid, K), order=(1, 0))
    w_d_block_ptr = tl.make_block_ptr(base=w_d_ptr, shape=(N, KB), strides=(KB, 1),
                                      offsets=(n_start, 0), block_shape=(n_per_pid, KB), order=(1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1),
                                    offsets=(0, n_start), block_shape=(M, n_per_pid), order=(1, 0))

    # step 1 (data preparation): decompression region degenerates to a load, because Q8_0 values need no decoding
    a_q = tl.load(a_q_block_ptr)                            # (M, K)          int8
    w_q = tl.load(w_q_block_ptr)                            # (n_per_pid, K)  int8

    # step 2 (matrix multiplication): expand_dims + broadcast_to to 3-D; step 3: cast to the product type, multiply
    a_q3 = tl.expand_dims(a_q, axis=1)                      # (M, 1, K)
    a_q3 = tl.broadcast_to(a_q3, (M, n_per_pid, K))         # (M, n_per_pid, K)
    w_q3 = tl.expand_dims(w_q, axis=0)                      # (1, n_per_pid, K)
    w_q3 = tl.broadcast_to(w_q3, (M, n_per_pid, K))         # (M, n_per_pid, K)
    prod = a_q3.to(tl.int16) * w_q3.to(tl.int16)            # (M, n_per_pid, K) int16

    # step 4.1.a (matrix multiplication): split-direction reshape; step 4.1.b: reduce the block dim in int32
    prod4 = tl.reshape(prod, (M, n_per_pid, KB, QK))        # (M, n_per_pid, KB, 32)
    blk = tl.sum(prod4.to(tl.int32), axis=3)                # (M, n_per_pid, KB) int32

    # step 4.1.c (scaling): apply the level-1 factors, variant 1 (both sides multiplied together first, then applied once)
    a_d = tl.load(a_d_block_ptr)                            # (M, KB)
    a_d3 = tl.expand_dims(a_d, axis=1)                      # (M, 1, KB)
    a_d3 = tl.broadcast_to(a_d3, (M, n_per_pid, KB))        # (M, n_per_pid, KB)
    w_d = tl.load(w_d_block_ptr)                            # (n_per_pid, KB)
    w_d3 = tl.expand_dims(w_d, axis=0)                      # (1, n_per_pid, KB)
    w_d3 = tl.broadcast_to(w_d3, (M, n_per_pid, KB))        # (M, n_per_pid, KB)

    d = a_d3.to(tl.float32) * w_d3.to(tl.float32)           # (M, n_per_pid, KB) f32
    lvl1 = blk.to(tl.float32) * d                           # (M, n_per_pid, KB) f32

    # step 5 (scaling): outermost plain reduction, no scale factor
    c = tl.sum(lvl1, axis=2)                                # (M, n_per_pid) float32

    tl.store(c_block_ptr, c)
```

## 11. Per-Format Difference Notes

The table instantiates the invariant of section 2 for the four target formats; the notes after it list only what differs from the verified Q8_0 × Q8_0 example of section 10. Runnable canonical-form kernels belong to WP A2 and will live under `benchmark/kernels/`; the iq2_xxs row is a plan rather than a result, and is marked as such because that format is not compiled and not numerically verified yet.

| Format | Paths and leaves | Levels: block size, factors, side | Merge |
|--------|------------------|-----------------------------------|-------|
| Q8_0 × Q8_0 | one path; leaves `a_q`, `w_q` | `L = 1`, `QK_1 = 32`; activation `a_d` and weight `w_d` | none |
| Q4_0 × Q8_0 | one path; leaves `a_q`, decoded `w_q` | `L = 1`, `QK_1 = 32`; activation `a_d` and weight `w_d` | none |
| Q4_K × Q8_K | two paths; main leaves `a_q`, `w_q`; zero-point leaf `a_q` | `{32, 256}`; level 1 weight `sc` on the main path, weight `m` on the zero-point path; level 2 weight `d` with activation `a_d` on the main path, weight `dmin` with activation `a_d` on the zero-point path | level 2, minus |
| iq2_xxs × Q8_K, not compiled, not numerically verified | one path; leaves `a_q`, codebook-decoded `w_q` | `{32, 256}`; level 1 the sub-block factor decoded on the weight side from `w_sc`; level 2 weight `w_d` and activation `a_d` | none |

**Q4_0 × Q8_0**

- Components: activation `a_q: int8 (m,k)` and `a_d: float16 (m,G_1)`; weight `w_q: uint8 (n,k/2)` and `w_d: float16 (n,G_1)`.
- Decompression: split each packed weight byte into low and high nibbles, subtract the static offset 8, and restore GGML's 32-value block order before producing `w_q: int8 (n,k)`. The activation side stays a direct Q8_0 load.

**Q4_K × Q8_K**

- Components: activation `a_q: int8 (m,k)`, `a_d: float32 (m,G_2)`; weight `w_q: uint8 (n,k/2)`, `w_d`, `w_dmin: float16 (n,G_2)`, and 6-bit `w_sc`, `w_m: uint8 (n,G_1)`.
- Level 2 merge, `main * (d * a_d) - zero * (dmin * a_d)`: each path takes its own level 2 factors before the subtraction. This form is numerically correct for any `d` and `dmin`, is the form WP A2 is to write, and is the form `benchmark/kernels/q4k_template_crosscheck.py` verifies.
- Level 1 merge, `(red1 * sc - bs1 * m) * (d * a_d)`: recognition accepts it structurally, because it satisfies factor invariance and carries equal cumulative quantization block size on both sides of the merge, but it equals the level 2 merge only when `d == dmin` and is numerically wrong otherwise.
- Decompression: unpack 4-bit weight codes with GGML's pairing across two adjacent 32-element sub-blocks — a byte's low nibble belongs to one sub-block and its high nibble to the next, not to the same sub-block — and unpack the 6-bit `w_sc` and `w_m` as factor-side decoding.

**iq2_xxs × Q8_K**

- Components: activation `a_q: int8 (m,k)`, `a_d: float32 (m,G_2)`; weight `w_idx`, `w_sign`, `w_sc`, `w_d`, plus GGML's codebook grid and sign table as two independent structures.
- Level 1 factor comes from factor-side decoding: the sub-block factor is derived from `w_sc` through a `(0.5 + s) * 0.25` computation instead of being loaded, while level 2 takes weight `w_d` and activation `a_d`.
- Decompression: load each 8-value codebook index, look up magnitudes in the codebook grid, decode signs through the separate sign table, and combine them without replacing the two structures by a signed codebook.

## 12. Measured Linalg Appendix

The operation groups below come from one real compilation of the verification kernel `benchmark/kernels/q80_q80_strict.py`, whose DAG is a single path with a single-level divisibility chain, so the `// step` annotations carry the step numbers of the single-path instance of section 2 and step 4.j instantiates as step 4.1. That instance has no merge point, so no addition or subtraction `linalg.generic` appears below; every other operation shape of the invariant does. Nothing in this appendix is hand-written or inferred, which makes it the baseline fact about the Linalg shape of each step of that instance.

```sh
source env.sh
rm -rf "$TRITON_CACHE_DIR"/*
rm -rf .triton/dump/shared
python benchmark/kernels/q80_q80_strict.py
ls .triton/dump/shared/
```

The compilation stage is `ttsharedir`, which runs `triton-shared-opt --triton-to-linalg-experimental` on the TTIR of the preceding stage; its output lands under `$TRITON_SHARED_DUMP_PATH`, by default `.triton/dump/shared`, as the file `ttshared.mlir`. The cache clear is not optional: a kernel already present in `$TRITON_CACHE_DIR` is served from the cache, so no stage runs and no dump appears. The dump holds the last specialization of the run, `M = 192`, `N = 128`, `K = 512`, `T = 4`, hence `n_per_pid = 32` and `G_1 = 16`. Read the shapes against those values: `192x32x512` is `(m, n, k)`, `192x32x16x32` is `(m, n, G_1, QK_1)`, `192x32x16` is `(m, n, G_1)`, and `192x32` is `(m, n)`. Every line below other than the `// step` annotations and the `// ... elided ...` markers is verbatim from that file; each marker stands for a removed region body. Step 1 has no group of its own here: with Q8_0 the region degenerates to a load, and the block pointer plus load lower into `memref.reinterpret_cast`, `memref.copy` and `bufferization.to_tensor` that carry no arithmetic.

```mlir
// step 2 — broadcast of the activation to (m, n, k); #map1 reads (d0, 0, d2), broadcastDims = 1
    %8 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<192x1x512xi16>) outs(%7 : tensor<192x32x512xi16>) attrs =  {broadcastDims = array<i64: 1>} {
// ... elided ...
// step 2 — broadcast of the weight to (m, n, k); #map2 reads (0, d1, d2), broadcastDims = 0
    %11 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x32x512xi16>) outs(%7 : tensor<192x32x512xi16>) attrs =  {broadcastDims = array<i64: 0>} {
// ... elided ...
// step 3 — three-dimensional element-wise multiply in the product type i16
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %11 : tensor<192x32x512xi16>, tensor<192x32x512xi16>) outs(%8 : tensor<192x32x512xi16>) {
// ... elided ...
// step 4.1.a — split-direction reshape (m, n, k) -> (m, n, G_1, QK_1)
    %expanded_3 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [192, 32, 16, 32] : tensor<192x32x512xi16> into tensor<192x32x16x32xi16> loc(#loc40)
// step 4.1.b — reduction along the innermost quantization block dimension, dimensions = [3]
    %reduced = linalg.reduce ins(%14 : tensor<192x32x16x32xi32>) outs(%16 : tensor<192x32x16xi32>) dimensions = [3] 
// ... elided ...
// step 4.1.c — the two sides' level-1 scale factors multiplied together
    %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %26 : tensor<192x32x16xf32>, tensor<192x32x16xf32>) outs(%23 : tensor<192x32x16xf32>) {
// ... elided ...
// step 4.1.c — the combined factor applied to the level-1 reduction result
    %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %27 : tensor<192x32x16xf32>, tensor<192x32x16xf32>) outs(%28 : tensor<192x32x16xf32>) {
// ... elided ...
// step 5 — outermost plain reduction with no scale factor, dimensions = [2]
    %reduced_10 = linalg.reduce ins(%29 : tensor<192x32x16xf32>) outs(%31 : tensor<192x32xf32>) dimensions = [2] 
// ... elided ...
```

The broadcasts of step 2 and of step 4.1.c each stay a `linalg.generic` of their own and write their result out to a full-size `(m, n, k)` or `(m, n, G_1)` intermediate tensor, instead of folding into the indexing map of the consumer that reads it. The cast to the accumulation type likewise stays a `linalg.generic` of its own instead of moving into the body of the `linalg.reduce` that follows it. Both foldings are loop scheduling decisions and belong to Phase B.

## 13. Open Items

The items below are deliberately left open, not oversights. Each names the work package that fixes it, and each waits on real cases rather than on a decision that could be taken now.

- The list of concrete rejection conditions, and the diagnostic wording each one carries, stays open until real non-conforming kernels exist to fix it against — WP A3, WP A5.
- Writing constraints inside the decompression region and inside factor-side decoding: section 6 describes both without enumerating what may appear in them, and the constraints tighten as formats accumulate — WP A2.
- The output form of layout feedback and of the cross-operator layout requirement, over the layout information section 7 makes recoverable — WP A3, WP A4.
- Divisibility chains of more than two levels: section 3 constructs a chain for any `L >= 1`, but only `L = 1` and `L = 2` are exercised by kernels in the tree — WP A2.
- Thread-level partitions other than the reference writing of section 8: a two-dimensional grid, a partition along M, and further partitions are accepted writings, since no legality condition mentions the grid, but no kernel in the tree exercises them, so they stay accepted and unverified — WP A2.
- One `ttshared.mlir` per format: WP A2's acceptance scope includes dumping that file for all four formats, because section 12 shows measured Linalg for the degenerate single-path format only. Capture is `source env.sh`, clear `$TRITON_CACHE_DIR`, run the kernel, take `ttshared.mlir` from `$TRITON_SHARED_DUMP_PATH` — WP A2.
- Q4_K's nibble pairing across two adjacent 32-element sub-blocks and iq2_xxs's separate codebook grid and sign table, both recorded in section 11, are awkward to write with the dimension transformations of section 6; which pairing and which table shape become the layout requirement put to the user is open — WP A2, WP B3 (Phase B).

## 14. Glossary and Scope Boundary

Every term below appears in the body of this document and carries the meaning fixed by `.kiro/steering/glossary.md`, which is authoritative for the project vocabulary; the reduction-DAG terms path, merge point and factor invariance are fixed by sections 2 and 4 here. The Chinese column gives the fixed equivalent, quoted for cross-reference with the proposal.

| Term | Chinese |
|------|---------|
| QGEMM, hierarchical scaling structure, canonical form | 量化矩阵乘算子, 层次缩放结构, 标准形式 |
| path, merge point, factor invariance | 路径, 合并点, 因子不变性 |
| quantization block, quantization block size, quantization level, divisibility chain, cumulative quantization block size | 量化块, 量化块大小, 量化级, 整除链, 累积量化块大小 |
| data preparation stage, matrix multiplication stage, scaling stage | 数据准备阶段, 矩阵乘阶段, 缩放阶段 |
| decompression region, factor-side decoding, planar component tensor, codebook | 解压区域, 因子侧解码, 分量张量, 码本 |
| scale factor, zero point, product type, accumulation type, scaled type | 缩放因子, 零点, 乘积类型, 累加类型, 缩放类型 |
| layout expression chain, layout feedback, cross-operator layout propagation | 布局表达链, 布局反馈, 跨算子布局传递 |
| thread-level partition, numeric contract, innermost tile size | 线程级任务划分, 数值契约, 最内层分块尺寸 |

Scope boundary. This document specifies QGEMM and does not cover GEMV, which is bound by memory bandwidth rather than by compute density. The project's compilation output ends at optimized Linalg IR, so VIR, the Vector dialect, LLVM IR and machine code are consumed from buddy-mlir rather than specified here. This document describes a user writing convention only: it fixes no compiler pass design and no implementation, and every place the text speaks of what the compiler does is an obligation on later work.

The compiler recognizes the canonical form and emits a diagnostic for input that does not conform, instead of exiting silently; the list of conditions it rejects on is open, see section 13.
