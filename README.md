# I-am-an-ai-engineer
- 송호연님이 출제해주신 문제 :)

--- 

- 요구사항1. Serverless API(Google Functions, Azure Functions, AWS Lambda)로 머신러닝 모델 CPU 서빙
	- Google Functions
- 요구사항2. 데이터셋은 Google BigQuery에 적재하고 꺼내서 사용
	- 이거야 쉬운데, pandas를 사용할듯
- 요구사항3. 학습 실험 관리 Opensource(Microsoft NNI, Google Adanet, Optuna 등)를 사용하여 AutoML 수행
	- Microsoft NNI 선택
- 요구사항4. 학습이 완료되면 Model Validation을 자동으로 수행해서 지금 서빙되고 있는 모델보다 우수한지 자동으로 검증
	- Meta 저장
- 요구사항5. 모델리스트가 관리되어야 하고, 선택적으로 배포 및 롤백이 가능함
	- experiment_id-trial_id
- 요구사항6. 모든 코드는 Pylint 가이드에 맞춰 깔끔함을 유지 (PEP8, Google Style 등)
    - pycharm 가이드라인을 모두 따름

- 서빙할 딥러닝 모델은 뭐든 상관없습니다. MNIST, 얼굴 인식, Object Detection

- 문제 푸는 기간: 2주
- 문제 제출 마감: 2019-09-08 14:00:00 (GMT+9)
- 제출 내용:
	- 1) Github repo link
	- 2) Serverless API Call example (Curl)


### Architecture
```
├── cloud_functions : cloud function 배포
│   ├── deploy.sh
│   ├── main.py
│   └── requirements.txt
├── compare_performance.py : 제일 좋은 성능(현재는 accuracy)을 저장
├── config
│   ├── best_model_metadata.json : 제일 좋은 성능을 가진 experiment_id, trial_id, params, performance 저장
│   ├── bigquery.yaml : bigquery 기본 설정 저장. 만약 실제 서비스였다면 staging/prod로 나눠야 함
│   └── train_search_space.json : nni에서 탐색할 파라미터들
├── data : 데이터 저장 폴더
├── data_loader.py 
├── log
│   └── model_output.log : 모델을 학습할 때마다 모델 experiment_id, trial_id, params 등을 저장
├── main.py
├── mnist_to_bigquery.py : mnist 데이터를 BigQuery로 저장, 불러옴
├── model_outputs : 실험-시도별로 모델 저장
│   ├── FwLMbXZz-JCT13-model.pth
│   └── ...
├── network.py : 네트워크(tensorflow / pytorch의 네트워크 둘 다 있음)
├── nni_config_tensorflow.yml 
├── nni_config_torch.yml
├── train_nni_tensorflow.py : tensorflow로 nni 실행 
├── train_nni_torch.py : pytorch로 nni 실행
└── utils.py : logger, config 존재
```

---

### 특징
- pd.read_gbq, pd.to_gbq를 활용해 빅쿼리와 통신했습니다
- nni의 experiment_id, trial_id, params를 매번 저장해 제일 좋은 모델을 확인합니다

---

### 후기 및 자기 회고
- 개인적으로 이런 Task들을 매우 즐겨하고 있어서, 파티 참석 여부와 관계없이 하려고 했습니다
- 파티 날짜를 보니 참석하기 어려울 것 같아, 안하고 있다가 어제 9월 7일에 시간 조금만 투자해서 코딩해봤습니다
- nni는 생각보다 꽤 심플하고 강력한 도구라 재미있었어요
- 요새 딥러닝 프레임워크를 자주 사용하지 않아서 까먹었네요. 추후 다시 튜토리얼 작성할 예정
- 이런 문제를 집중있게 구현해보는거 매우 유익! 준비해주신 송호연님 감사합니다
- 이런 부분으로 스터디를 더 하고 싶은 생각 가득-!

---

### TODO(언제 할지.. 모르지만..)
- Serving API, google functions 배포랑 조절하는거 선택 기능
    - 왠지 모르겠지만 람다 deploy에서 오류 생김 => 코드 문제일듯
- 테스트 코드
- accuracy가 제일 높다고 항상 좋은 모델은 아니기 때문에, 모델을 선택하는 기준을 좀 더 다양하게 가져갈 수 있도록 => TFDV 참고
    - 모델을 선택적으로 서빙할 수 있도록 


---

### 명령어(main.py)
- nni를 통해 train

    ```
    python3 main.py --mode train
    ```
    
- 학습 후 최고 성능의 모델 정보 저장하기

    ```
    python3 main.py --mode find_best_model
    ```
    
- cloud function에 배포하기

    ```
    python3 main.py --mode serving
    ```
    
- mnist 데이터를 BigQuery에 저장하기(1번만 실행하면 됨)

    ```
    python3 main.py --mode upload_mnist
    ```
    
- BigQuery의 mnist 데이터를 로컬로 저장하기

    ```
    python3 main.py --mode donwload_mnist
    ```

---

### 명령어(수동)
- nni pytorch 실행

    ```
    nnictl create --config nni_config_torch.yml
    ```
    
- nni 실행 포트 죽이기(8080일 경우)

	```
	kill -9 $(lsof -t -i:8080)
	```    

---

### nni 실행 화면
<img src="https://www.dropbox.com/s/7p2wivprky9r5vz/Screenshot%202019-09-08%2015.12.58.png?raw=1">


---
	
### Reference
- [정태환님 Github](https://github.com/graykode/mnist-flow) : 제일 빠르게 제출해주셔서 많은 영감을 받게 해주신!
- [김준태님 Github](https://github.com/OPAYA/Model_Serving) : 저랑 구조가 동일한데 이미지까지 첨부해주시고 더 친절한!
- [송호연님 Brunch](https://brunch.co.kr/@chris-song/91) : 문제 출제해주시고, 튜토리얼도 작성해주신 호연님!
- [김경선님 Github](https://github.com/llable/AnoIn-BigQuery) : BigQuery 튜토리얼을 작성해주신!