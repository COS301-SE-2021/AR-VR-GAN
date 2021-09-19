import { Observable } from 'rxjs';

export interface ModelGeneration {
  generateImage(Request: Observable<RequestProxy>): Observable<any>;
  loadModel(Request: RequestModel): RequestModelResponse;
  listModels(Request: RequestListModel): any;
  currentModel(Request: RequestCurrentModel): any;
  trainModel(Request: RequestTrainModel): any;
}

export interface RequestProxy {
  vector: number[];
}

export interface RequestModel {
  modelName: string;
}

export interface RequestModelResponse {
  succesful: boolean;
  message:string;
}

export interface RequestTrainModelResponse {
  succesful: boolean;
  message:string;
}

export interface RequestListModel {
  default: boolean;
  saved: boolean;
}

export interface RequestCurrentModel {

}

export interface RequestTrainModel {
  modelName: string;
  trainingEpochs: number;
  latentSize: number;
  datasetName: string;
  beta: number;
  modelType: string;
}