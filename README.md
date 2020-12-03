# 열차동적모델 README

- 코드 목록
  - `metroLSTM_feature_engineering.py`: 신분당선 열차차상데이터에서 학습용 데이터를 만들어주는 코드.  
    사용 예시: `python metroLSTM_feature_engineering.py --predictstep=5`  
    
  ```
    --predictstep PREDICTSTEP
                        choose the predicted step: for example, 1, 10, 30, 50,
                        etc.
  ```
  - `metroLSTM_dynamic_model.py`: 학습용 신분당선 열차차상데이터를 가지고서 LSTM 모델 학습 및 속도 예측 그래프 산출에 쓰이는 코드.  
    사용 예시: `python metroLSTM_dynamic_model.py --predictstep=5 --explore_hp=0 --hs=26 --lr=0.0001 --bs=32`  
    
  ```
  --gpu GPU             Turn GPU on(GPU number) or off(-1). Default is -1.
  --predictstep PREDICTSTEP
                        Choose the predicted step: for example, 1, 10, 30, 50,
                        etc. Default value is 10.
  --activation ACTIVATION
                        Choose the recurrent activation function: "sigmoid" or
                        "mish". Default is mish.
  --explore_hp EXPLORE_HP
                        Turn the parameter search on(1) or off(0). Default is
                        1.
  --hs HS               Determine the hidden unit size of model. This option
                        is valid only when explore_hp is 0.
  --lr LR               Determine the learning rate of model. This option is
                        valid only when explore_hp is 0.
  --bs BS               Determine the batch size of model. This option is
                        valid only when explore_hp is 0.
  ```

  - `MetroLSTMCore.py`: `'metro'`가 붙은 파일에서 쓰이는 주요 클래스를 담아둔 코드.  
  - `MetroLSTMconfig.py`: `metroLSTM_dynamic_model.py`의 하이퍼패러미터 탐색 범위 등의 설정을 담아둔 코드. 하이퍼패러미터 탐색 시 이 파일 내부의 값을 조정하여 범위를 설정.  
  
- 데이터는 계약에 의거하여 GitHub 저장소 내에서는 삭제처리되었음.
