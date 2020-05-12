def xgboost(xTrain,yTrain,xTest):
  params = {'max_depth': 8,
            'eta': 0.5,
            'silent': 1,
            'min_child_weight':0.5,
            'seed':1,
            'num_round': 1000,
            'eval_metric':'rmse'}
  params = {'max_depth': 8,
            'eta': 0.5,
            'silent': 1,
            'min_child_weight':0.5,
            'seed':1,
            'num_round': 1000,
            'eval_metric':'rmse'}
  xgbMatrix = xgb.DMatrix(xTrain, yTrain)
  xgbTrain = xgb.train(params, xgbMatrix)
  yTrainPredicted = xgbTrain.predict(xgbMatrix)
  
  xgb.plot_importance(xgbTrain)
  
  yTestPredicted = xgbTrain.predict(xgb.DMatrix(xTest))
  print(yTestPredicted)
  
  yTestPredicted = list(map(lambda i: max(0, i), list(yTestPredicted)))
  print(pd.DataFrame(yTestPredicted).describe())
  
  return yTrainPredicted