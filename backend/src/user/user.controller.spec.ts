import { Test, TestingModule } from '@nestjs/testing';
import { UserController } from './user.controller';
import { UserService } from './user.service';

describe('UserController', () => {
  let controller: UserController;

  const mockUserService = {
    registerUser: jest.fn((dto) => {
      return {
        id: Date.now(),
        ...dto
      }
    }),
    updateUserWithId: jest.fn((id, dto) => {
      return {
        id,
        ...dto
      }
    })
  }

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UserController],
      providers: [UserService]
    }).overrideProvider(UserService).useValue(mockUserService).compile();

    controller = module.get<UserController>(UserController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  it('should register a user', () => {
    const dto = {username: 'jason', email: 'jman89412@gmail.com', password: 'test123'}

    expect(controller.registerUser(dto)).toEqual({
      id: expect.any(Number),
      username: dto.username,
      email: dto.email,
      password: dto.password
    });
  });

  it('should update a user', () => {
    const dto = {username: 'jason', email: 'jman89412@gmail.com', password: 'test123'}

    expect(controller.updateUserWithId('12415352', dto)).toEqual({
      id: '12415352',
      ...dto
    })
  });
});
