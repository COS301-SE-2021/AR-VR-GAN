import { Observable } from 'rxjs';

export interface ModelGeneration {
  generateImage(Request: Observable<RequestDto>): Observable<any>;
}

interface RequestDto {
  data: number[];
}