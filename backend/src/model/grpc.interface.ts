import { Observable } from 'rxjs';

export interface ModelGeneration {
  generateImage(numberArray: RequestDto): Observable<any>;
}

interface RequestDto {
  data: number[];
}