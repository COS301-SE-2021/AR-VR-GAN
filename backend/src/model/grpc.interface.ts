import { Observable } from 'rxjs';

export interface ModelGeneration {
  generateImage(Request: Observable<RequestProxy>): Observable<any>;
  LoadModel(Request: RequestModel): any;
  ListModels(Request: RequestListModel): any;
  CurrentModel(): any;
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