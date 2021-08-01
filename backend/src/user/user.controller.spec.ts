import { Test, TestingModule } from '@nestjs/testing';
import { UserController } from './user.controller';
import { UserService } from './user.service';
import { MockUserService } from './mocks/user.mock'
import { RegisterUserDto } from './dto/register-user.dto';
import { GetUserByUsernameDto } from './dto/get-user-by-username.dto';
import { UpdateUserByUsernameDto } from './dto/update-user-by-username.dto';

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
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");

    expect(controller.registerUser(registerDto)).toEqual({
      username: registerDto.username,
      email: registerDto.email,
      password: registerDto.password
    });
  });

  // it('should update a user', () => {
  //   const dto = {username: 'jason', email: 'jman89412@gmail.com', password: 'test123'}

  //   expect(controller.updateUserWithUsername('12415352', dto)).toEqual({
  //     id: '12415352',
  //     ...dto
  //   })
  // });

  // it('Get all users', () => {

  //   expect(controller.getAllUsers()).toEqual("User List")
  // });

  it('Get user by the username', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    const dto = new GetUserByUsernameDto("jwtToken","test123");
    expect(controller.getUserByUsername(dto)).toEqual({
      success: true,
      message: dto.username,
      user: dto.jwtToken
    })
  });

  // it('delete a user by a username', () => {
  //   const id = '12345';
  //   const out = id + " Deleted"
  //   expect(controller.deleteUserByUsername(id)).toEqual(out)
  // });

});
