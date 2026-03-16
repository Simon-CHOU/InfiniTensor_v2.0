#pragma once
#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "core/graph.h"
#include "core/op_type.h"
#include "operators/ElementWise.h"
#include "operators/Gemm.h"
#include "operators/Conv.h"
#include "operators/LayerNorm.h"

namespace infini {

class GraphBuilderObj {
  private:
    Ref<GraphObj> g;

  public:
    GraphBuilderObj(Runtime runtime);

    Tensor tensor(ShapeExpr dims, DataType dtype,
                  std::optional<StrideExpr> stride = std::nullopt);

    Tensor gemm(Tensor A, Tensor B, Tensor C, float alpha = 1.0,
                float beta = 1.0, bool transA = false, bool transB = false,
                std::optional<Tensor> Y = std::nullopt);
    Tensor add(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor sub(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor mul(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor clip(Tensor input, Tensor min, Tensor max,
                std::optional<Tensor> output = std::nullopt);
    
    Tensor conv(Tensor input, Tensor weight, std::optional<Tensor> bias,
                std::vector<int> pads, std::vector<int> strides,
                std::vector<int> dilations, std::optional<Tensor> output = std::nullopt);

    Tensor layer_norm(Tensor input, Tensor weight, Tensor bias, float eps = 1e-5,
                      std::optional<Tensor> output = std::nullopt);

    string printGraph() const;

    Graph getGraph() const;
};

} // namespace infini
#endif // GRAPH_BUILDER_H
