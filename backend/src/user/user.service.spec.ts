import { Test, TestingModule } from '@nestjs/testing';
import { UserService } from './user.service';
import { MockUserModel } from './mocks/userRepository.mock'
import { UserSchema } from './schemas/user.schema'
import { getModelToken } from '@nestjs/mongoose';

describe('UserService', () => {
  let service: UserService;



  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ 
        UserService,
        {
          provide : getModelToken('User'),
          useValue : MockUserModel
        }],
    }).compile();

    service = module.get<UserService>(UserService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should register a user', async () => {
    const dto = {username: 'jason', email: 'jman89412@gmail.com', password: 'test123'}

    expect(await service.registerUser(dto)).toEqual({
      username: dto.username,
      email: dto.email,
      password: dto.password
    });
  });

});
