import { RegisterUserDto } from '../dto/register-user.dto';
import { GetUserByUsernameDto, GetUserByUsernameResponse } from '../dto/get-user-by-username.dto';
import { UserResponse } from '../dto/user-response.dto';
import { GetAllUsersDto, GetAllUsersResponse } from '../dto/get-all-users.dto';
import { userDTO } from './userInterface.mock';
import { UsersModule } from '../user.module';
import { LoginUserDto } from '../dto/login-user.dto';
import { UpdateUserByUsernameDto } from '../dto/update-user-by-username.dto';

export const MockUserService = {
    registerUser: jest.fn((dto) => {
      return {
        ...dto
      }
    }),

    loginUser:jest.fn((username,password) => {
      const MockUsers = [
        {username: "username1", password: "password1"},
        {username: "username2", password: "password2"},
        {username: "username3", password: "password3"},
      ]

      
      
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









export default class MockUserClass {
 //private users:Array<userDTO>;
 private users:userDTO[] = new Array(10)

  public registerUser(Registerdto){
    let user = new userDTO(Registerdto.username,Registerdto.password,Registerdto.email);
    if((user.username!="")){
      if(user.password!=""){
        if(user.email!=""){
          this.users.push(user);
          let res = new UserResponse(true, 'The user was registered successfully.');
          return res;
        }
        else{
          let res = new UserResponse(false, 'No email entered');
          return res;
        }
      }
      else{
        let res = new UserResponse(false, 'No password entered');
        return res;
      }
    }
    else{
      let res = new UserResponse(false, 'No username entered');
      return res;
    }
  }
  

  public loginUser(loginDto){
    let user = new LoginUserDto(loginDto.username,loginDto.password);
    for (var ik=0 ; ik < this.users.length ; ik++)
    {
      if (this.users[ik] != null)
      {
        if(this.users[ik].username == user.username && this.users[ik].password == user.password)
        {
            let res = new UserResponse(true, 'login succesful!');
            return res;
        }
        else
        {
          let res = new UserResponse(false, 'login fail!');
          return res;
        }
      }
    }
  }

  public updateUserWithUsername(updateDto){
    let user = new UpdateUserByUsernameDto(updateDto.jwtToken,updateDto.currentUsername,updateDto.newUsername,updateDto.newPassword,updateDto.newEmail);
    for (var ik=0 ; ik < this.users.length ; ik++)
    {
      if (this.users[ik] != null)
      {
        if(this.users[ik].username == user.currentUsername)
        {
          this.users[ik] = new userDTO(updateDto.newUsername,updateDto.newPassword,updateDto.newEmail);
          let res = new UserResponse(true, 'updated succesfully!');
          return res;
        }
      }
    }

    let res = new UserResponse(false, 'updated failed! user not found');
    return res;
  }

 


  public getUserByUsername(getUserByUsernameDto){
    let user = new GetUserByUsernameDto(getUserByUsernameDto.jwtToken,getUserByUsernameDto.username);
    console.log(getUserByUsernameDto.username);
    for (var ik=0 ; ik < this.users.length ; ik++)
    {
      if (this.users[ik] != null)
      {
        if(this.users[ik].username == user.username)
        {
          let res = new GetUserByUsernameResponse(true, getUserByUsernameDto.username, getUserByUsernameDto);
          return res;
        }
      }
    }
    let res = new GetUserByUsernameResponse(false, "getUserByUsernameDto.username", getUserByUsernameDto);
    return res;
  }

 public getAllUsers(getAllUsersdto){
    let user = new GetAllUsersDto(getAllUsersdto.jwtToken);
    let res = new GetAllUsersResponse(true, 'The list of all users is attatched.', getAllUsersdto);
    return res;

  }

}
