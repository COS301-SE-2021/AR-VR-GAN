import { Observable } from 'rxjs';

export interface ModelGeneration {
  generateImage(Request: Observable<RequestProxy>): Observable<any>;
  LoadModel(Request: RequestModel): any;
}

export interface RequestProxy {
  vector: number[];
}

export interface RequestModel {
  modelName: string;
}