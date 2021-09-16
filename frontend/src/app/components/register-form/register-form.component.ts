import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import {FormControl, FormGroupDirective, NgForm, Validators} from '@angular/forms';
import { ErrorStateMatcher } from '@angular/material/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Router } from '@angular/router';
import { HOST_URL } from 'src/config/consts';

@Component({
  selector: 'app-register-form',
  templateUrl: './register-form.component.html',
  styleUrls: ['./register-form.component.css']
})
export class RegisterFormComponent implements OnInit {
  hideFirst: boolean = true;
  hideSecond: boolean = true;

  usernameFormControl = new FormControl('', [
    Validators.required
  ]);

  passwordFormControl = new FormControl('', [
    Validators.required
  ]);

  passwordConfirmFormControl = new FormControl('', [
    Validators.required
  ]);

  emailFormControl = new FormControl('', [
    Validators.required
  ]);

  matcher = new MyErrorStateMatcher();
  
  constructor(
    private http: HttpClient, 
    private snackBar: MatSnackBar,
    private router: Router) { }

  ngOnInit(): void {
  }

  registerUser(username: string, email:string, password: string, passwordConfirmation: string) {    
    if (passwordConfirmation != password) {
      this.snackBar.open("Please ensure that your confirmation password matches your actual password.", "Close");
      return;
    }

    if (password.length < 8) {
      this.snackBar.open("Please ensure that your password is at least 8 characters long.","Close");
      return;
    }

    if (username.length < 4) {
      this.snackBar.open("Please ensure that your username is at least 4 characters long.","Close");
      return;
    }

    if ((email.length < 3) || (!email.includes(".")) || (!email.includes("@"))) {
      this.snackBar.open("Please enter a valid email address.", "Close");
      return;
    }

    this.http.post<any>(HOST_URL + '/user/register/', {'username': username, 'email': email, 'password': password}).subscribe(resp => {
      this.snackBar.open(resp.message, "Close");

      if (resp.success) {
        this.router.navigate(['/login']);
      }
    });
  }
}

export class MyErrorStateMatcher implements ErrorStateMatcher {
  isErrorState(control: FormControl | null, form: FormGroupDirective | NgForm | null): boolean {
    const isSubmitted = form && form.submitted;
    return !!(control && control.invalid && (control.dirty || control.touched || isSubmitted));
  }
}