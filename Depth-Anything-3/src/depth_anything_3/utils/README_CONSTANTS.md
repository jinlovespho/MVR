# Constants 설정 가이드 (Collaborative Workflow)

## 문제 상황
여러 사람이 같이 작업할 때 `constants.py`의 `DA3_LQ_ROOT_PATH`와 `DA3_RES_ROOT_PATH`를 수정하면 git merge conflict가 발생합니다.

## 해결 방법: 환경변수 사용

`constants.py`는 이제 **환경변수**에서 경로를 읽습니다. 환경변수가 설정되어 있으면 그 값을 사용하고, 없으면 기본값을 사용합니다.

### 사용 방법

#### 1단계: 자신의 bash 설정 파일에 추가

`.bashrc`, `.zshrc` 또는 `.bash_profile`에 다음을 추가하세요:

```bash
# 본인의 경로로 설정
export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100'
```

#### 2단계: Shell 재시작 또는 source 명령 실행

```bash
source ~/.bashrc
# 또는
source ~/.zshrc
```

#### 3단계: 확인

```bash
echo $DA3_LQ_ROOT_PATH
echo $DA3_RES_ROOT_PATH
```

### 예시

**사용자 A:**
```bash
export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100'
```

**사용자 B:**
```bash
export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_200'
export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_200'
```

### 기본값 (환경변수 미설정 시)

| 변수 | 기본값 |
|------|--------|
| `DA3_LQ_ROOT_PATH` | `/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_200` |
| `DA3_RES_ROOT_PATH` | `/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100` |

### Python에서 환경변수 직접 설정 (선택사항)

Python 스크립트에서 실행 전에 설정할 수도 있습니다:

```python
import os
os.environ['DA3_LQ_ROOT_PATH'] = '/your/custom/path'
os.environ['DA3_RES_ROOT_PATH'] = '/your/custom/path'

# 그 다음 constants import
from depth_anything_3.utils.constants import DA3_LQ_ROOT_PATH, DA3_RES_ROOT_PATH
```

## 장점

✅ **No more merge conflicts**: 각 사용자가 자신의 bash 설정에서만 수정  
✅ **공유 가능한 constants.py**: git에 commit해도 안전  
✅ **간단한 설정**: 환경변수만 설정하면 됨  
✅ **기본값 지원**: 환경변수 미설정 시에도 작동  

## 참고

- `constants.py`의 모든 상수는 여전히 `from depth_anything_3.utils.constants import ...`로 import 가능합니다.
- 환경변수는 프로세스별로 독립적이므로 서로 다른 터미널에서는 서로 다른 설정을 사용할 수 있습니다.
