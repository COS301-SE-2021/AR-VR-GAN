import { Test, TestingModule } from '@nestjs/testing';
import { UserController } from './user.controller';
import { UserService } from './user.service';
import MockUserClass, { MockUserService } from './mocks/user.mock'
import { RegisterUserDto } from './dto/register-user.dto';
import { GetUserByUsernameDto } from './dto/get-user-by-username.dto';
import { UpdateUserByUsernameDto } from './dto/update-user-by-username.dto';
import { GetAllUsersDto, GetAllUsersResponse } from './dto/get-all-users.dto';
import { LoginUserDto } from './dto/login-user.dto';


describe('UserController', () => {
  let controller: UserController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UserController],
      providers: [UserService]
    }).overrideProvider(UserService).useClass(MockUserClass).compile();

    controller = module.get<UserController>(UserController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });



  it('should register a user', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");

    expect(controller.registerUser(registerDto)).toEqual({
      success:true,
      message: 'The user was registered successfully.'
    });
  });


  it('register-false : no username', () => {
    const registerDto = new RegisterUserDto("","test123@test.com","test123");

    expect(controller.registerUser(registerDto)).toEqual({
      success:false,
      message: 'No username entered'
    });
  });

  it('register-false : no email', () => {
    const registerDto = new RegisterUserDto("test123","","test123");

    expect(controller.registerUser(registerDto)).toEqual({
      success:false,
      message: 'No email entered'
    });
  });

  it('register-false : no password', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","");

    expect(controller.registerUser(registerDto)).toEqual({
      success:false,
      message: 'No password entered'
    });
  });








  it('login a user', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    let loginDto =  new LoginUserDto("test123","test123")
    expect(controller.loginUser(loginDto)).toEqual({
      success: true,
      message: "login succesful!"
    })
  });

  it('should detect an incorrect login', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    let loginDto =  new LoginUserDto("test12","test123")
    expect(controller.loginUser(loginDto)).toEqual({
      success: false,
      message: "login fail!"
    })
  });


  

  it('should update a user', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    let updateDto = new UpdateUserByUsernameDto("jwtToken","test123","newUser","newPass","newEmail");

    expect(controller.updateUserWithUsername(updateDto)).toEqual({
      success: true,
      message: "updated succesfully!"
    })
  });



  it('should fail to update user', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    let updateDto = new UpdateUserByUsernameDto("jwtToken","test12","newUser","newPass","newEmail");

    expect(controller.updateUserWithUsername(updateDto)).toEqual({
      success: false,
      message: "updated failed! user not found"
    })
  });





  it('Get all users', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    let allUserDto = new GetAllUsersDto("jwtToken")
    expect(controller.getAllUsers(allUserDto)).toEqual({
      success: true,
      message: "The list of all users is attatched.",
      users: "jwtToken"
    })
  });

  it('Get user by the username', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    let dto = new GetUserByUsernameDto("jwtToken","test123");
    expect(controller.getUserByUsername(dto)).toEqual({
      success: true,
      message: dto.username,
      user: null
    })
  });

  it('Should not be able to find the user,User isnt registered', () => {
    let dto = new GetUserByUsernameDto("jwtToken","test123");
    expect(controller.getUserByUsername(dto)).toEqual({
      success: false,
      message: "could not find user",
      user: null
    })
  });

  it('delete a user by a username', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)

    let deleteDto =  new GetUserByUsernameDto("jwtToken","test123")
    expect(controller.deleteUserByUsername(deleteDto)).toEqual({
      success: true,
      message: "The user has been removed."
    })
  });

  it('Should not be able to delete the user', () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    controller.registerUser(registerDto)
    
    let deleteDto =  new GetUserByUsernameDto("jwtToken","test")
    expect(controller.deleteUserByUsername(deleteDto)).toEqual({
      success: false,
      message: "There is no user with the given username."
    })
  });

});
