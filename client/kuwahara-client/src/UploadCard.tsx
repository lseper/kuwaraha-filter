import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Card from "@mui/material/Card";
import Container from "@mui/material/Container";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import { useState } from "react";
import BeforeAndAfterImages from "./BeforeAndAfterImages";

function UploadCard() {
	const [selectedFile, setSelectedFile] = useState<File | undefined>(
		undefined
	);

	const [returnedFile, setReturnedFile] = useState<File | undefined>(
		undefined
	);

	const [kernelSize, setKernelSize] = useState(20);

	const [previewURL, setPreviewURL] = useState<string | undefined>(undefined);
	const [outputURL, setOutputURL] = useState<string | undefined>(undefined);

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files && event.target.files[0];
		if (file) {
			setSelectedFile(file);

			const reader = new FileReader();
			reader.onload = (e) => {
				if (typeof e.target?.result === "string") {
					setPreviewURL(e.target.result);
				}
			};
			reader.readAsDataURL(file);
		}
	};

	const handleUpload = async () => {
		if (selectedFile) {
			const formData = new FormData();
			formData.append("file", selectedFile, selectedFile.name);
			formData.append("kernel", `${kernelSize}`);
			const res = await fetch("http://127.0.0.1:5000/filter", {
				method: "POST",
				body: formData,
				mode: "cors",
				headers: {
					"Access-Control-Allow-Origin": "*",
				},
			});

			console.log("AFTER RESPONSE");
			if (res.ok) {
				console.log("ASFDHISADHNFGADSF");
				// const form = await res.formData();
				const img = await res.blob();
				if (img) {
					const reader = new FileReader();
					reader.onload = (e) => {
						if (typeof e.target?.result === "string") {
							setOutputURL(e.target.result);
						}
					};
					reader.readAsDataURL(img);
				}
			}
		}
	};

	return (
		<Container
			sx={{
				display: "flex",
				flexDirection: "column",
				justifyContent: "center",
				alignItems: "center",
			}}
		>
			<Card variant="outlined" sx={{ minWidth: 275, maxWidth: 500 }}>
				<CardContent>
					<Stack spacing={2}>
						<Typography
							sx={{
								display: "flex",
								flexDirection: "row",
								justifyContent: "center",
								alignItems: "center",
							}}
							variant="h3"
							gutterBottom
						>
							Try it yourself!
						</Typography>
						<Button
							variant="contained"
							component="label"
							color="secondary"
						>
							Select File
							<input
								type="file"
								accept=".png, .PNG, .jpeg, .jpg, .JPG, .JPEG"
								hidden
								name="file"
								onChange={handleFileChange}
							/>
						</Button>
						<Typography gutterBottom>Kernel Size</Typography>
						<Slider
							aria-label="Kernel Size"
							defaultValue={20}
							valueLabelDisplay="auto"
							value={kernelSize}
							step={2}
							min={2}
							onChange={(e, nv) =>
								setKernelSize(
									typeof nv === "number" ? nv : nv[0]
								)
							}
							max={100}
						/>
					</Stack>
				</CardContent>
				<CardActions
					sx={{
						display: "flex",
						flexDirection: "row",
						justifyContent: "center",
						alignItems: "center",
					}}
				>
					<Button
						variant="contained"
						color="primary"
						onClick={handleUpload}
					>
						Apply
					</Button>
				</CardActions>
			</Card>
			{previewURL && (
				<BeforeAndAfterImages
					beforeImageUrl={previewURL}
					afterImageUrl={outputURL}
				/>
			)}
		</Container>
	);
}

export default UploadCard;
