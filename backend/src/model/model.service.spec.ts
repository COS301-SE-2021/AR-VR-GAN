import { Test, TestingModule } from '@nestjs/testing';
import { model, Model } from 'mongoose';
import { MockModelService } from './mocks/model.mock';
import { ModelService } from './model.service';

describe('ModelService', () => {
  let service: ModelService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ModelService, {
        provide: ModelService,
        useValue: {MockModelService}
      }],
    }).compile();

    service = module.get<ModelService>(ModelService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should sum up the data', () => {
    const dto = { data : [1,2,3] }
    let sum = 0;

    for (let i = 0; i < dto.data.length; i++) {
        sum += dto.data[i]
    }

    expect(service.handleCoords(dto)).toEqual(sum);
  });
});
