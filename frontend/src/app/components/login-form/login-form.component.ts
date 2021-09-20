import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MatSnackBar } from '@angular/material/snack-bar';
import { CookieService } from 'ngx-cookie-service';
import { Router } from '@angular/router';
import { HOST_URL } from 'src/config/consts';
import { FormControl, FormGroupDirective, NgForm, Validators } from '@angular/forms';
import { ErrorStateMatcher } from '@angular/material/core';
import { AppComponent } from 'src/app/app.component';

@Component({
  selector: 'app-login-form',
  templateUrl: './login-form.component.html',
  styleUrls: ['./login-form.component.css']
})
export class LoginFormComponent implements OnInit {
  hide: boolean = true;
  message: string = '';

  usernameFormControl = new FormControl('', [
    Validators.required
  ]);

  passwordFormControl = new FormControl('', [
    Validators.required
  ]);

  matcher = new MyErrorStateMatcher();

  constructor(
    private http: HttpClient, 
    private snackBar: MatSnackBar, 
    private cookieService: CookieService,
    private router: Router,
    private appComponent: AppComponent) { }

  ngOnInit(): void {
  }

  loginUser(username: string, password: string): void {
    this.message = '';
    this.cookieService.deleteAll();

    this.http.post<any>(HOST_URL + '/user/login/', {'username': username, 'password': password}).subscribe(resp => {
      if (resp.success) {
        this.message = 'The user was logged in successfully.';

        this.cookieService.set('username', username);
        this.cookieService.set('jwtToken', resp.message);
        this.appComponent.loggedIn = true;
        this.router.navigate(['']);
      } else {
        this.message = resp.message;
      }

      this.snackBar.open(this.message, "Close");
    });
  }
}

export class MyErrorStateMatcher implements ErrorStateMatcher {
  isErrorState(control: FormControl | null, form: FormGroupDirective | NgForm | null): boolean {
    const isSubmitted = form && form.submitted;
    return !!(control && control.invalid && (control.dirty || control.touched || isSubmitted));
  }
}