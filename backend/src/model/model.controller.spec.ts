import { Test, TestingModule } from '@nestjs/testing';
import { ModelController } from './model.controller';
import { MockModelService } from './mocks/model.mock';
import { ModelService } from './model.service';
import { request } from 'express';
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

  it('should sum up the data', () => {
    const dto = { data : [1,2,3] }
    let sum = 0;

    for (let i = 0; i < dto.data.length; i++) {
        sum += dto.data[i]
    }

    expect(controller.handleCoords(dto)).toEqual({sum : sum});
  });

});
