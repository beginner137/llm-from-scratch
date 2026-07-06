import triton
import triton.language as tl
import torch
from einops import rearrange


@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_tr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_tr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    """
    i = 0
    row0 [ 0  1  2 ]  3  4  5   6  7
    row1 [ 8  9 10 ] 11 12 13  14 15

    i = 1
    row0  0  1  2  [ 3  4  5 ]  6  7
    row1  8  9 10  [11 12 13 ] 14 15

    i = 2
    row0  0  1  2   3  4  5  [ 6  7  pad0 ]
    row1  8  9 10  11 12 13  [14 15  pad0 ]
    """

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(
            0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(
            0,), padding_option="zero")

        output += tl.sum(row*weight[None, :], axis=1)
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D = x.shape[-1]

        # reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)

        assert len(
            weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        n_rows = x.shape[0]
        y = torch.empty((n_rows,), device=x.device)
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])


def weighted_sum(x, weight):
    return WeightedSumFunc.apply(x, weight)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("triton_weighted_sum.py requires CUDA")

    x = torch.arange(12, device="cuda", dtype=torch.float32).view(2, 3, 2)
    weight = torch.tensor([10.0, 20.0], device="cuda")
    y = weighted_sum(x, weight)
    expected = torch.sum(x * weight, dim=-1)

    print("x:")
    print(x)
    print("weight:")
    print(weight)
    print("y:")
    print(y)
    print("expected:")
    print(expected)
