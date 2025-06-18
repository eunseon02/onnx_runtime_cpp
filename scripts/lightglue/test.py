import torch
from lightglue_wrapper import LightGlueWrapper

# 1) 모델 생성
model = LightGlueWrapper(**LightGlueWrapper.default_conf).eval()

# 2) TorchScript 로 스크립트
#    (리턴값 타입이 dict 이므로, 스크립팅 시 예시 출력값을 타입 힌트에 맞춰 줍니다.)
example_args = (
    torch.randn(1,1,480,640),
    torch.randn(1,382,2),
    torch.randn(1,382,256),
    torch.randn(1,1,480,640),
    torch.randn(1,382,2),
    torch.randn(1,382,256),
)
traced = torch.jit.trace(model, example_args, strict=False)
torch.jit.save(traced, "lightglue_ts.pt")