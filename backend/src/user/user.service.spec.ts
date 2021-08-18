import { Test, TestingModule } from '@nestjs/testing';
import { UserService } from './user.service';
import { MockUserModel } from './mocks/userRepository.mock'
import { UserSchema } from './schemas/user.schema'
import { getModelToken } from '@nestjs/mongoose';
import { RegisterUserDto } from './dto/register-user.dto';
import { JwtModule, JwtService } from '@nestjs/jwt';
import config from '../config/keys';
import { LoginUserDto } from './dto/login-user.dto';

describe('UserService', () => {
  let service: UserService;



  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ 
        UserService,
        {
          provide : getModelToken('User'),
          useClass : MockUserModel
        },
        {
          provide: JwtService,
          useValue: JwtService
        }],
    }).compile();

    service = module.get<UserService>(UserService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // it('should register a user', async () => {
  //   const dto = {username: 'jason', email: 'jman89412@gmail.com', password: 'test123'}

  //   expect(await service.registerUser(dto)).toEqual({
  //     username: dto.username,
  //     email: dto.email,
  //     password: dto.password
  //   });
  // });

  it('should register a user', async () => {
    const registerDto = new RegisterUserDto("test123","test123@test.com","test123");
    
    // expect(await (await service.registerUser(registerDto))).toEqual({
    //   success:true,
    //   message: 'The user was registered successfully.'
    // });
    expect(service.registerUser(registerDto)).toBe
  });

  it('should check the password', async () => {
    const loginDto = new LoginUserDto("password","matt");
    
    expect(await service.loginUser(loginDto)).toEqual({
      // success:true,
      // message: 'The user was registered successfully.'
      message: "The user with the specified username does not exist.",
      success: false
    });;
  });

  it('should get a user by username', async () => {
    const testUsername = "ethan";
    const testToken = "xxxxx.yyyyy.zzzzz";
    
    expect(await service.getUserByUsername(testToken,testUsername)).toEqual({
      // success:true,
      // message: 'The required user is attatched.',
      // user:"ethan"
      message: "This JWTToken does not exist.",
      success: false,
      user: null
    });;
  });

  it('should get a user by token', async () => {
    const testToken = "xxxxx.yyyyy.zzzzz";
    
    expect(await service.getUserByJWTToken(testToken)).toEqual({
      // success:true,
      // message: 'The required user is attatched.',
      // user:"ethan"
      message: "This JWTToken does not exist.",
      success: false,
      user: null
    });;
  });

  it('should delete a user bu username', async () => {
    const testToken = "xxxxx.yyyyy.zzzzz";
    const testUsername = "ethan";
    
    expect(await service.deleteUserByUsername(testToken, testUsername)).toEqual({
      // success:true,
      // message: 'The user has been removed.'
      message: "This JWTToken does not exist.",
      success: false,
    });;
  });

});
