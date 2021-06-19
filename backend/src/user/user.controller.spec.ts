import { Test, TestingModule } from '@nestjs/testing';
import { UserController } from './user.controller';
import { UserService } from './user.service';
import { MockUserService } from './mocks/user.mock'

describe('UserController', () => {
  let controller: UserController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UserController],
      providers: [UserService]
    }).overrideProvider(UserService).useValue(MockUserService).compile();

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

  it('Get all users', () => {

    expect(controller.getAllUsers()).toEqual("User List")
  });

  it('Get all users', () => {
    const id = '12345';
    const out = id + " Found"
    expect(controller.getUserById(id)).toEqual(out)
  });
  it('Get all users', () => {
    const id = '12345';
    const out = id + " Deleted"
    expect(controller.deleteUserById(id)).toEqual(out)
  });

});
