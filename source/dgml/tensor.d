module dgml.tensor;

import mir.ndslice;

/// A tensor wrapper around a Mir Slice
struct Tensor(size_t N)
{
    /// The underlying data
    Slice!(float*, N) data;

    /// Constructor from existing slice
    this(Slice!(float*, N) s)
    {
        data = s;
    }

    /// Allocate a new tensor with given shape
    static Tensor!N allocate(size_t[N] shape...)
    {
        import mir.ndslice.allocation : slice;
        return Tensor!N(slice!float(shape));
    }

    /// Shape property
    auto shape() const @property
    {
        return data.shape;
    }
}

/// Matrix multiplication: C = A * B
/// A is [M x K], B is [K x N], C is [M x N]
Tensor!2 matmul(Tensor!2 a, Tensor!2 b)
{
    assert(a.shape[1] == b.shape[0], "Dimension mismatch for matmul: columns of A must match rows of B");

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1]; // using N to match standard notation

    auto c = Tensor!2.allocate(M, N);
    
    // Naive implementation
    // Initialize to 0 is handled by slice allocation? No, slice!float uses uninitialized or default init.
    // basic types are default initialized to nan for float? 
    // Wait, D floats default to nan. We need to zero it.
    c.data[] = 0.0f;

    foreach (i; 0 .. M)
    {
        foreach (j; 0 .. N)
        {
            float sum = 0.0f;
            foreach (k; 0 .. K)
            {
                sum += a.data[i, k] * b.data[k, j];
            }
            c.data[i, j] = sum;
        }
    }

    return c;
}

/// Tiled Matrix multiplication
Tensor!2 matmul_tiled(Tensor!2 a, Tensor!2 b, size_t blockSize = 32)
{
    assert(a.shape[1] == b.shape[0], "Dimension mismatch for matmul: columns of A must match rows of B");

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    auto c = Tensor!2.allocate(M, N);
    c.data[] = 0.0f;

    for (size_t i = 0; i < M; i += blockSize)
    {
        for (size_t j = 0; j < N; j += blockSize)
        {
            for (size_t k = 0; k < K; k += blockSize)
            {
                // Process block
                size_t iMax = (i + blockSize > M) ? M : i + blockSize;
                size_t jMax = (j + blockSize > N) ? N : j + blockSize;
                size_t kMax = (k + blockSize > K) ? K : k + blockSize;

                for (size_t ii = i; ii < iMax; ++ii)
                {
                    for (size_t jj = j; jj < jMax; ++jj)
                    {
                        for (size_t kk = k; kk < kMax; ++kk)
                        {
                            c.data[ii, jj] += a.data[ii, kk] * b.data[kk, jj];
                        }
                    }
                }
            }
        }
    }
    return c;
}

unittest
{
    import std.stdio;
    
    // Naive Test
    auto a = Tensor!2.allocate(2, 3);
    a.data[0, 0] = 1; a.data[0, 1] = 2; a.data[0, 2] = 3;
    a.data[1, 0] = 4; a.data[1, 1] = 5; a.data[1, 2] = 6;

    auto b = Tensor!2.allocate(3, 2);
    b.data[0, 0] = 7; b.data[0, 1] = 8;
    b.data[1, 0] = 9; b.data[1, 1] = 10;
    b.data[2, 0] = 11; b.data[2, 1] = 12;

    auto c = matmul(a, b);
    assert(c.data[0, 0] == 58);
    assert(c.data[0, 1] == 64);
    assert(c.data[1, 0] == 139);
    assert(c.data[1, 1] == 154);
    
    // Tiled Test (using small block size to force tiling)
    auto c_tiled = matmul_tiled(a, b, 2);
    assert(c_tiled.data[0, 0] == 58);
    assert(c_tiled.data[0, 1] == 64);
    assert(c_tiled.data[1, 0] == 139);
    assert(c_tiled.data[1, 1] == 154);

    writeln("Matmul tests passed!");
}
