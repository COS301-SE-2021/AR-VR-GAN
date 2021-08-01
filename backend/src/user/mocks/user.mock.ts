import { RegisterUserDto } from '../dto/register-user.dto';
import { GetUserByUsernameResponse } from '../dto/get-user-by-username.dto';
import { UserResponse } from '../dto/user-response.dto';
import { GetAllUsersDto, GetAllUsersResponse } from '../dto/get-all-users.dto';

export const MockUserService = {
    registerUser: jest.fn((dto) => {
      return {
        ...dto
      }
    }),

    loginUser:jest.fn((username,password) => {
      let resp = new UserResponse(true,"login succesful!")
      return resp
    }),

    updateUserWithUsername: jest.fn((dto) => {
      let resp = new UserResponse(true,"updated succesfully!")
      return resp
    }),

    getAllUsers: jest.fn((dto) => {
      let resp = new GetAllUsersResponse(true,"all users list",dto)
      return resp
    }),

    getUserByUsername:jest.fn((token,username) => {
      let resp = new GetUserByUsernameResponse(true,username,token)
      return resp;
    }),

    deleteUserByUsername:jest.fn((id) => {
      let resp = new UserResponse(true,"deleted succesfully!")
      return resp
    }),
  }
