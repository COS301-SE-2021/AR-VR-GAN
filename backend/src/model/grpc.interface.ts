import { Observable } from 'rxjs';

export interface ModelGeneration {
  generateImage(Request: Observable<RequestProxy>): Observable<any>;
  loadModel(Request: RequestModel): any;
  listModels(Request: RequestListModel): any;
  currentModel(Request: RequestCurrentModel): any;
}

export interface RequestProxy {
  vector: number[];
}

export interface RequestModel {
  modelName: string;
}

export interface RequestListModel {
  default: boolean;
  saved: boolean;
}

export interface RequestCurrentModel {

}