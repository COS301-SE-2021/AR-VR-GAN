import { ReplaySubject } from "rxjs";
import { RequestProxy } from "../grpc.interface";

export const MockModelService= {

    handleCoords: jest.fn((dto) => {
        let sum = 0;

        for (let i = 0; i < dto.data.length; i++) {
            sum += dto.data[i]
        }
        return sum;
    }),

    proxy: jest.fn((request) => {
        const subject = new ReplaySubject<RequestProxy>();
        subject.next({ vector: request.data });
        subject.complete();
        const stream = MockModelService.grpcService.generateImage(subject.asObservable());
        return stream.toPromise();
    })
  }
