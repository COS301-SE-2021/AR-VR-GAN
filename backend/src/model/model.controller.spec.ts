import { Test, TestingModule } from '@nestjs/testing';
import { ModelController } from './model.controller';
import { MockModelService } from './mocks/model.mock';
import { Request } from './interfaces/request.interface';
import { ModelService } from './model.service';
import { Observable, ReplaySubject, Subject } from 'rxjs';

describe('ModelController', () => {
  let controller: ModelController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [ModelController],
      providers: [ModelService]
    }).overrideProvider(ModelService).useValue(MockModelService).compile();

    controller = module.get<ModelController>(ModelController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  // it('should sum up the data', () => {
  //   const dto = { data : [1,2,3] }
  //   let sum = 0;

  //   for (let i = 0; i < dto.data.length; i++) {
  //       sum += dto.data[i]
  //   }

  //   expect(controller.handleCoords(dto)).toEqual({sum : sum});
  // });

  it('should be proxied', () => {
    const subject = new ReplaySubject<Request>();
    subject.next({ data: [1,2,3] });
    subject.complete();
    controller.proxy(subject).subscribe(
      res => res.data.toEqual(6)
    );
  });

});
