import { Observable } from 'rxjs';

export interface ModelGeneration {
  generateImage(Request: Observable<RequestProxy>): Observable<any>;
}

export interface RequestProxy {
  vector: number[];
}