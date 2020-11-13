# rncTrainLSTM

- 파일 목록
  - `metroLSTM_feature_engineering.py`: `csv` 파일에서 `pkl` 데이터를 만들어주는 코드.  
    사용 예시: `python metroLSTM_feature_engineering.py --predictstep=1`
  - `metroLSTM_dynamic_model.py`: `pkl` 파일을 가지고서 LSTM 모델 학습 및 속도 예측 그래프 산출에 쓰이는 코드.  
    사용 예시: `python metroLSTM_dynamic_model.py --predictstep=1 --explore_hp=0 --hs=10 --lr=0.0001 --bs=128`  
    
  ```
  --gpu GPU             Turn GPU on(GPU number) or off(-1). Default is -1.
  --predictstep PREDICTSTEP
                        Choose the predicted step: 1, 10, 30, 50, 100. Default
                        value is 10.
  --activation ACTIVATION
                        Choose the activation function instead of mish:
                        sigmoid, swish.
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
  
    
  - `MetroLSTMCore.py`: 다른 파일에서 쓰이는 주요 클래스를 담아둔 코드. 
