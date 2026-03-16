#include "core/graph_builder.h"

namespace infini {

GraphBuilderObj::GraphBuilderObj(Runtime runtime)
    : g(make_ref<GraphObj>(std::move(runtime))) {}

Tensor GraphBuilderObj::tensor(ShapeExpr dims, DataType dtype,
                               std::optional<StrideExpr> stride) {
    if (stride.has_value()) {
        return g->addTensor(dims, stride.value(), dtype);
    } else {
        return g->addTensor(dims, dtype);
    }
}

Tensor GraphBuilderObj::gemm(Tensor A, Tensor B, Tensor C, float alpha,
                             float beta, bool transA, bool transB,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<GemmObj>(std::move(A), std::move(B),
                                     std::move(Y.value()), std::move(C), alpha,
                                     beta, transA, transB);
        return Y.value();
    } else {
        return g
            ->addOp<GemmObj>(std::move(A), std::move(B), nullptr, std::move(C),
                             alpha, beta, transA, transB)
            ->getOutput(0);
    }
}

#define DEFINE_BINARY_OP(OP, TYPE)                                             \
    Tensor GraphBuilderObj::OP(Tensor A, Tensor B, std::optional<Tensor> Y) {  \
        if (Y.has_value()) {                                                   \
            g->addOpWithOutputs<ElementWiseObj>(                               \
                TYPE, std::move(A), std::move(B), std::move(Y.value()));       \
            return Y.value();                                                  \
        } else {                                                               \
            return g                                                           \
                ->addOp<ElementWiseObj>(TYPE, std::move(A), std::move(B),      \
                                        nullptr)                               \
                ->getOutput(0);                                                \
        }                                                                      \
    }

DEFINE_BINARY_OP(add, OpType::Add);
DEFINE_BINARY_OP(sub, OpType::Sub);
DEFINE_BINARY_OP(mul, OpType::Mul);

Tensor GraphBuilderObj::clip(Tensor input, Tensor min, Tensor max,
                             std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<ElementWiseObj>(OpType::Clip, std::move(input),
                                            std::move(min), std::move(max),
                                            std::move(output.value()));
        return output.value();
    } else {
        return g
            ->addOp<ElementWiseObj>(OpType::Clip, std::move(input),
                                    std::move(min), std::move(max), nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::conv(Tensor input, Tensor weight, std::optional<Tensor> bias,
                             std::vector<int> pads, std::vector<int> strides,
                             std::vector<int> dilations, std::optional<Tensor> output) {
    Tensor b = bias.has_value() ? bias.value() : nullptr;
    if (output.has_value()) {
        g->addOpWithOutputs<ConvObj>(std::move(input), std::move(weight),
                                     std::move(output.value()), std::move(pads),
                                     std::move(strides), std::move(dilations), std::move(b));
        return output.value();
    } else {
        return g->addOp<ConvObj>(std::move(input), std::move(weight), nullptr,
                                 std::move(pads), std::move(strides), std::move(dilations), std::move(b))
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::layer_norm(Tensor input, Tensor weight, Tensor bias, float eps,
                                   std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<LayerNormObj>(std::move(input), std::move(weight), std::move(bias),
                                          std::move(output.value()), eps);
        return output.value();
    } else {
        return g->addOp<LayerNormObj>(std::move(input), std::move(weight), std::move(bias), nullptr, eps)
            ->getOutput(0);
    }
}

string GraphBuilderObj::printGraph() const { return g->toString(); }

Graph GraphBuilderObj::getGraph() const { return g; }
} // namespace infini
