import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn_crfsuite import CRF, metrics
from collections import Counter
import warnings
import os
import shutil


class FibroPred:
    def __init__(self, years = 0, typem = "CRF"):
        self.seed = 42
        self.model_type = typem
        self.years = years
        self.targets = ['Progressive disease', 'Death', 'Necessity of transplantation']
        self.X_train, self.X_test, self.y_train, self.y_test = self._select_cols()
        #self.global_info = self._clustering()

        if self.model_type == "CRF":
            self.models, self.feature_importances = self._find_models()
        else:
          self.models = self._find_models()

        #self.global_expl = self.global_explainability()
    
    def _crf_analysis(self, feature_importances, target):

        def print_state_features(state_features):
            for (attr, label), weight in state_features:
                print("%0.6f %-8s %s" % (weight, label, attr))


        crf = self.models[target]
        print("Top positive:")
        print_state_features(Counter(crf.state_features_).most_common(5))

        print("\nTop negative:")
        print_state_features(Counter(crf.state_features_).most_common()[-5:])

        feature_importance = feature_importances[target]
        print(f"Feature importances for {target}:")
        # Sort the feature importances by absolute value of coefficients
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, importance in sorted_features[:5]:  # Show top 10 features
            print(f"Feature: {feature}, Importance: {importance}")


    def inference(self, row):
        row = row.drop(['COD NUMBER', 'Transplantation Status', 'Necessity of transplantation', 'Progressive disease', 'Death'], errors='ignore')
        # Specify the columns to drop
        columns_to_drop = ['COD NUMBER', 'Transplantation Status', 'Necessity of transplantation', 
                          'Progressive disease', 'Death']

        # Drop the specified columns from the original DataFrame
        df_dropped_columns = row.drop(columns=columns_to_drop, errors='ignore')
        print(df_dropped_columns.columns)

        # Create a new DataFrame with the selected row and the remaining columns
        new_df = pd.DataFrame(row, columns=df_dropped_columns.columns)

        for col in new_df.select_dtypes(include='object').columns:
            new_df[col] = new_df[col].astype('category')

        if self.model_type != "CRF":
            predictions, paths = self.local_explainability(new_df)
        else:
            pass 
        return predictions, paths
        



    def local_explainability(self, ddata):
        preds = {}
        paths = []
        output_dir = "tmp"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        for target in self.targets:
            print("tarfet", type(target), target)
            model = self.models[target]       
            predictions = model.predict(ddata)
            print("Prediction of ", target, predictions)
            explainer = shap.Explainer(self.models[target])
            shap_values = explainer(ddata)

            plt.figure()          
            shap.plots.waterfall(shap_values[0], show = False)
            print("predictions", predictions)
            preds[str(target)] = predictions[0]
            name_waterfall = "waterfall_plot" + target + ".png"
            plt.savefig(os.path.join(output_dir, name_waterfall), bbox_inches="tight", dpi=300)
            plt.close()
            plt.figure()
            shap.force_plot(shap_values[0], matplotlib=True, show = False)
            name_force = "force_plot"+target+".png"
            plt.savefig(os.path.join(output_dir, name_force), bbox_inches="tight", dpi=300) 
            plt.close()
            paths.append([os.path.join(output_dir, name_force),os.path.join(output_dir, name_waterfall)])
        return preds, paths

    


    def _do_shap(self, target):
        print(self.models.keys())
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(self.models[target])
        shap_values = explainer.shap_values(self.X_test)


        output_dir = "tmp"
        if not os.path.exists(output_dir):
              os.makedirs(output_dir)
                    
        


        shap_values = explainer(self.X_test)  # Get Explanation object

        output_dir = "tmp"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate and save bar plot
        plt.figure()
        shap.plots.bar(shap_values, show = False)
        bar_name = "bar_plot"+target+".png"
        plt.savefig(os.path.join(output_dir, bar_name), bbox_inches="tight", dpi=300)
        plt.close()

        # Generate and save beeswarm plot
        plt.figure
        shap.plots.beeswarm(shap_values, show = False)
        bee_name =  "beeswarm_plot"+target+".png"
        plt.savefig(os.path.join(output_dir,bee_name), bbox_inches="tight", dpi=300)
        plt.close()
        




    def global_explainability(self):
        for target in self.targets:
            if self.model_type == "CRF":
                self.analysis = self._crf_analysis(self.feature_importances, target)
            else:
              self._do_shap(target)
        # retrun les grafiques
            #self._clustering(target)
            


    def _select_cols(self):
  
        #df = df.drop(columns=['ProgressiveDisease', 'Necessity of transplantation','Cause of death','Date of death'])
       

        if self.years == 0:
            df = pd.read_csv("content/FibroPredCODIFICADA_Updated_after_diagnosis.csv", sep=";")
            
        
        if self.years == 1:
            df = pd.read_csv("content/FibroPredCODIFICADA_Updated_after_1years.csv",sep =";")

        if self.years == 2:
            df = pd.read_csv("content/FibroPredCODIFICADA_Updated_after_2years.csv", sep =";")

        df['Death'] = df['Death'].fillna(0)
        df['Progressive disease'] = df['Progressive disease'].fillna(0)
        df['Necessity of transplantation'] = df['Transplantation Status'].fillna(0)
        df = df.drop('COD NUMBER', axis=1)
        df = df.drop('Transplantation Status', axis=1)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')


        X = df.drop(columns=self.targets)
        y = df[self.targets]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=self.seed)

        return X_train, X_test, y_train, y_test

    def _get_best_model(self, target):
        param_grid = {'n_estimators': [50, 100, 200],'max_depth': [3, 5, 7],'learning_rate': [0.01, 0.1, 0.2],'subsample': [0.8, 1.0]}

        grid_search = GridSearchCV(
          estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, 
          enable_categorical=True,),
          param_grid=param_grid,
          scoring='roc_auc',
          cv=3,
          verbose=1)
        grid_search.fit(self.X_train, self.y_train[target]) 
        print("Best Parameters:", grid_search.best_params_)

        best_model = XGBClassifier(
            **grid_search.best_params_,
            use_label_encoder=False,
            eval_metric='logloss',
            enable_categorical=True
        )

       

        best_model.fit(self.X_train, self.y_train[target]) 
        print(target)
        #xgboost.plot_importance(best_model, importance_type='weight')
        #xgboost.to_graphviz(best_model, num_trees=0)

        y_pred = best_model.predict(self.X_test)
        print(f"Accuracy for {target}:", accuracy_score(self.y_test['Progressive disease'], y_pred))
        print(f"Classification Report for {target}:\n{classification_report(self.y_test[target], y_pred)}")
        print(f"ROC AUC {target}:", roc_auc_score(self.y_test['Progressive disease'], y_pred))

        return best_model
        

    def _crfmodel(self,target, feature_importances):

        def row_to_features(row):
            return {col: str(row[col]) for col in row.index}

        X_train_seq = [[row_to_features(row) for _, row in self.X_train.iterrows()]]
        y_train_seq = [[str(label) for label in self.y_train[target]]]

        X_test_seq = [[row_to_features(row) for _, row in self.X_test.iterrows()]]
        y_test_seq = [[str(label) for label in self.y_test[target]]]

        # Define and fit the CRF model
        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train_seq, y_train_seq)
        feature_importances[target] = crf.state_features_

        # Make predictions
        y_pred = crf.predict(X_test_seq)

        # Evaluate the CRF model
        print(metrics.flat_classification_report(y_test_seq, y_pred))
        
        return crf, feature_importances

    def _find_models(self):
          models = {}
          feature_importances = {}
        
          output_dir = "models_pesos"
          if not os.path.exists(output_dir):
              os.makedirs(output_dir)

          for target in self.targets:
              print("Training model for target", target)
              if self.model_type != "CRF":
                  name_model = "xgboost_" + target + "_" + str(self.years) + ".json"
                  path_model = os.path.join(output_dir,name_model)

                  if os.path.exists(path_model):
                      print("Loading models")
                      model = XGBClassifier(enable_categorical=True)  # Create an empty XGBClassifier
                      model.load_model(path_model)  # Load the model using XGBoost's load_model method
                      models[target] = model
                  else:
                      model = self._get_best_model(target)
                      models[target] = model
                      model.save_model(path_model)
              else:
                  model, feature_importances = self._crfmodel(target, feature_importances)
                  models[target] = model

          if self.model_type != "CRF":
              return models
          else:
            return models, feature_importances


warnings.filterwarnings('ignore')
'''a = FibroPred(years = 0, typem = "patata")'''