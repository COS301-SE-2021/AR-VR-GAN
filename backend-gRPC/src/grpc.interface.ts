import { Observable } from 'rxjs';

export interface IGrpcService {
  handleCoords(numberArray: RequestDto): Observable<any>;
}

interface RequestDto {
  data: number[];
}
